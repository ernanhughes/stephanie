# stephanie/agents/maintenance/scorable_embedding_backfill.py
from __future__ import annotations

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.scorable_embedding_store import ScorableEmbeddingStore
from stephanie.models.casebook import CaseORM
from stephanie.models.document import DocumentORM
from stephanie.models.hypothesis import HypothesisORM
from stephanie.models.plan_trace import PlanTraceORM
from stephanie.models.prompt import PromptORM
from stephanie.scoring.scorable import ScorableFactory


class ScorableEmbeddingBackfillAgent(BaseAgent):
    ORM_MAP = {
        "document": DocumentORM,
        "prompt": PromptORM,
        "response": PromptORM,
        "hypothesis": HypothesisORM,
        "case": CaseORM,
        "plan_trace": PlanTraceORM
    }

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # self.scorable_type = cfg.get("scorable_type", "case")
        self.scorable_type = cfg.get("scorable_type", "response")
        self.embed_full_document = cfg.get("embed_full_document", True)
        self.embedding_type = self.memory.embedding.name  # e.g. "hf_embeddings"

        if self.scorable_type not in self.ORM_MAP:
            raise ValueError(f"Unsupported scorable_type: {self.scorable_type}")

        self.orm_cls = self.ORM_MAP[self.scorable_type]

    async def run(self, context: dict) -> dict:
        updated, skipped = 0, 0

        # Step 1: Fetch all documents of the given 
        scorables  = []
        with self.memory.session() as session:
            scorables = session.query(self.orm_cls).all()
        total_docs = len(scorables)

        # Wrap in tqdm progress bar
        for scorable in tqdm(scorables, desc=f"Backfilling {self.scorable_type} embeddings", unit="doc"):
            # Step 2: Check if embedding already exists in the embedding store
            exists = self.memory.scorable_embeddings.get_by_scorable(
                scorable_id=str(scorable.id),
                scorable_type=self.scorable_type,
                embedding_type=self.embedding_type,
            )
            if exists:
                skipped += 1
                continue 

            # Step 3: Choose text for embedding
            scorable = ScorableFactory.from_orm(scorable, mode="response_only")

            # Step 4: Generate embedding

            # Step 5: Insert into store
            embedding_id = self.memory.scorable_embeddings.get_or_create(scorable)

            self.logger.log("ScorableEmbeddingBackfilled", {
                "scorable_id": str(scorable.id),
                "scorable_type": self.scorable_type,
                "embedding_id": embedding_id,
                "embedding_type": self.embedding_type,
            })

            updated += 1

        # Final log + report
        summary = {
            "event": self.name,
            "document_type": self.document_type,
            "embedding_type": self.embedding_type,
            "updated": updated,
            "skipped": skipped,
            "total": total_docs,
        }

        self.report(summary) 
        context[self.output_key] = summary
        return context
