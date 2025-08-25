# stephanie/agents/maintenance/document_embedding_backfill.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.document import DocumentORM
from stephanie.models.prompt import PromptORM
from stephanie.models.hypothesis import HypothesisORM
from stephanie.memory.document_embedding_store import DocumentEmbeddingStore
from stephanie.scoring.scorable_factory import ScorableFactory
from tqdm import tqdm


class DocumentEmbeddingBackfillAgent(BaseAgent):
    ORM_MAP = {
        "document": DocumentORM,
        "prompt": PromptORM,
        "hypothesis": HypothesisORM,
    }

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.store = DocumentEmbeddingStore(memory.session, logger=logger)
        self.document_type = cfg.get("document_type", "document")
        self.embed_full_document = cfg.get("embed_full_document", True)
        self.embedding_type = self.memory.embedding.name  # e.g. "hf_embeddings"

        if self.document_type not in self.ORM_MAP:
            raise ValueError(f"Unsupported document_type: {self.document_type}")

        self.orm_cls = self.ORM_MAP[self.document_type]

    async def run(self, context: dict) -> dict:
        session = self.memory.session
        updated, skipped = 0, 0

        # Step 1: Fetch all documents of the given type
        documents = session.query(self.orm_cls).all()
        total_docs = len(documents)

        # Wrap in tqdm progress bar
        for doc in tqdm(documents, desc=f"Backfilling {self.document_type} embeddings", unit="doc"):
            # Step 2: Check if embedding already exists in the embedding store
            exists = self.memory.document_embeddings.get_by_document(
                document_id=str(doc.id),
                document_type=self.document_type,
                embedding_type=self.embedding_type,
            )
            if exists:
                skipped += 1
                continue

            # Step 3: Choose text for embedding
            scorable = ScorableFactory.from_orm(doc, mode="full")

            # Step 4: Generate embedding
            embedding_vector = self.memory.embedding.get_or_create(scorable.text)
            embedding_id = self.memory.embedding.get_id_for_text(scorable.text)

            # Step 5: Insert into store
            if embedding_id:
                self.store.insert({
                    "document_id": str(doc.id),
                    "document_type": self.document_type,
                    "embedding_id": embedding_id,
                    "embedding_type": self.embedding_type,
                })

                self.logger.log("DocumentEmbeddingBackfilled", {
                    "document_id": str(doc.id),
                    "document_type": self.document_type,
                    "embedding_id": embedding_id,
                    "embedding_type": self.embedding_type,
                })

                updated += 1

        session.commit()

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
