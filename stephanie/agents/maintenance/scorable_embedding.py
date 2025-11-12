# stephanie/agents/maintenance/scorable_embedding.py
from __future__ import annotations

from dataclasses import asdict

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.text_item import TextItem
from stephanie.models.casebook import CaseORM
from stephanie.models.chat import ChatTurnORM
from stephanie.models.document import DocumentORM
from stephanie.models.hypothesis import HypothesisORM
from stephanie.models.plan_trace import PlanTraceORM
from stephanie.models.prompt import PromptORM
from stephanie.scoring.scorable import ScorableFactory, ScorableType
from stephanie.utils.progress_mixin import ProgressMixin


class ScorableEmbeddingAgent(BaseAgent, ProgressMixin):
    ORM_MAP = {
        "document": DocumentORM,
        "prompt": PromptORM,
        "response": PromptORM,
        "hypothesis": HypothesisORM,
        "case": CaseORM,
        "plan_trace": PlanTraceORM,
        "conversation_turn": ChatTurnORM
    }

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # self.scorable_type = cfg.get("scorable_type", "case")
        self.scorable_type = cfg.get("scorable_type", ScorableType.CONVERSATION_TURN)
        self.embed_full_document = cfg.get("embed_full_document", True)
        self.embedding_type = self.memory.embedding.name  # e.g. "hf_embeddings"

        if self.scorable_type not in self.ORM_MAP:
            raise ValueError(f"Unsupported scorable_type: {self.scorable_type}")

        self.orm_cls = self.ORM_MAP[self.scorable_type]
        self.limit = int(cfg.get("limit", 1000))
        self.ab_enabled = bool(cfg.get("ab_enabled", False))
        pre = cfg.get("preload", {}) or {}
        self.include_full_text = bool(cfg.get("include_full_text", True))
        self.include_ner = bool(cfg.get("include_ner", True))
        self.preload_enabled = bool(pre.get("enabled", self.ab_enabled))
        self.preload_total = int(pre.get("total_limit", max(self.limit * 10, 1000)))
        self.preload_include_texts = bool(pre.get("include_texts", True))
        self.preload_require_ner = bool(pre.get("require_nonempty_ner", False))
        self.preload_require_assistant = bool(pre.get("require_assistant_text", False))
        self.preload_min_len = int(pre.get("min_assistant_len", 0))
        self.preload_order_desc = bool(pre.get("order_desc", True))


    async def run(self, context: dict) -> dict:
        updated, skipped = 0, 0
        task = f"ScorableLoad:{context.get('pipeline_run_id', 'na')}"
        self._init_progress(self.container, self.logger)

        # Step 1: Fetch all documents of the given 
        scorables  = []
        if self.scorable_type == ScorableType.CONVERSATION_TURN:
            # Choose fetch profile: oversample if enabled, otherwise use the tight/defaults
            fetch_total = self.preload_total if self.preload_enabled else self.limit
            fetch_include_texts = self.preload_include_texts if self.preload_enabled else self.include_full_text
            fetch_require_nonempty_ner = self.preload_require_ner if self.preload_enabled else self.include_ner
            fetch_require_assistant = self.preload_require_assistant if self.preload_enabled else True
            fetch_min_len = self.preload_min_len if self.preload_enabled else 1
            fetch_order_desc = self.preload_order_desc if self.preload_enabled else True

            self.pstart(
                task=task,
                total=fetch_total,
                meta={
                    "target_type": str(self.scorable_type),
                    "fetch_total": fetch_total,
                    "batch_size": self.batch_size,
                    "preload_enabled": self.preload_enabled,
                    "filters": {
                        "require_assistant_text": fetch_require_assistant,
                        "require_nonempty_ner": fetch_require_nonempty_ner,
                        "min_assistant_len": fetch_min_len,
                    },
                },
            )

            produced = 0
            for batch in self.memory.chats.iter_turns_with_texts(
                total_limit=fetch_total,
                batch_size=self.batch_size,
                include_texts=fetch_include_texts,
                include_goal=True,
                require_assistant_text=fetch_require_assistant,
                require_nonempty_ner=fetch_require_nonempty_ner,
                min_assistant_len=fetch_min_len,
                order_desc=fetch_order_desc,
            ):
                for row in batch:
                    scorables.append(asdict(TextItem.from_chat_turn(row)))
                produced += len(batch)
                self.ptick(task=task, done=produced, total=fetch_total)
                if produced >= fetch_total:
                    break

            self.pdone(task=task)
        else:
            with self.memory.session() as session:
                scorables = session.query(self.orm_cls).all()
        total_docs = len(scorables)

        # Wrap in tqdm progress bar
        for scorable in tqdm(scorables, desc=f"Backfilling {self.scorable_type} embeddings", unit="doc"):
            # Step 2: Check if embedding already exists in the embedding store
            exists = self.memory.scorable_embeddings.get_by_scorable(
                scorable_id=str(scorable.get("id")),
                scorable_type=self.scorable_type,
                embedding_type=self.embedding_type,
            )
            if exists:
                self.ptick(task=task, done=produced, total=fetch_total)
                skipped += 1
                continue 

            # Step 3: Choose text for embedding
            scorable = ScorableFactory.from_dict(scorable)

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
            "scorable_type": self.scorable_type,
            "embedding_type": self.embedding_type,
            "updated": updated,
            "skipped": skipped,
            "total": total_docs,
        }

        self.report(summary) 
        context[self.output_key] = summary
        return context
