# stephanie/agents/knowledge/knowledge_db_loader.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, PIPELINE_RUN_ID


class KnowledgeDBLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.top_k = cfg.get("top_k", 10)
        self.include_full_text = cfg.get("include_full_text", False)
        self.target_type = cfg.get(
            "target_type", "document"
        )  # or "section"
        self.include_ner = cfg.get("include_ner", False)

        self.doc_ids_scoring = cfg.get("doc_ids_scoring", False)
        self.doc_ids = cfg.get("doc_ids", [])

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get("goal_text", "")
        pipeline_run_id = context.get(PIPELINE_RUN_ID)

        # 1. Fetch documents
        if self.doc_ids_scoring:
            if not self.doc_ids:
                self.logger.log(
                    "NoDocumentIdsProvided", "No document ids to score."
                )
                return context

            docs = self.memory.documents.get_by_ids(self.doc_ids)
            if not docs:
                self.logger.log("NoDocumentsFound", {"ids": self.doc_ids})
                return context
            self.logger.log(
                "DocumentsLoadedByIds",
                {"count": len(docs), "ids": self.doc_ids},
            )
            docs = [d.to_dict() for d in docs]
        else:
            docs = self.memory.embedding.search_related_scorables(
                goal_text, top_k=self.top_k, include_ner=self.include_ner, target_type=self.target_type
            )
            self.logger.log(
                "DocumentsSearched",
                {
                    "count": len(docs),
                    "goal_text": goal_text,
                    "top_k": self.top_k,
                },
            )

        # 2. Save retrieved doc dicts into context
        context[self.output_key] = docs

        context["retrieved_ids"] = [d["id"] for d in docs]

        for d in docs:
            self.memory.pipeline_references.insert(
                {
                    "pipeline_run_id": pipeline_run_id,
                    "scorable_type": d["scorable_type"],
                    "scorable_id": d["scorable_id"],
                    "relation_type": "retrieved",
                    "source": self.name,
                }
            )

        self.logger.log(
            "KnowledgeDBLoaded",
            {"count": len(docs), "search_method": self.target_type},
        )

        return context
