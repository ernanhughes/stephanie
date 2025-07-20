# stephanie/agents/knowledge/knowledge_db_loader.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class KnowledgeDBLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.top_k = cfg.get("top_k", 100)
        self.include_full_text = cfg.get("include_full_text", False)
        self.search_method = cfg.get(
            "search_method", "document"
        )  # or "section"
        self.doc_ids_scoring = cfg.get("doc_ids_scoring", False)
        self.doc_ids = cfg.get("doc_ids", [])   

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get("goal_text", "")


        if self.doc_ids_scoring:
            if not self.doc_ids:
                self.logger.log("NoDocumentIdsProvided", "No document ids to score.")
                return context

            docs = self.memory.document.get_by_ids(self.doc_ids)
            if not docs:
                self.logger.log("NoDocumentsFound", {"ids": self.doc_ids})
                return context
            self.logger.log(
                "DocumentsLoadedByIds",
                {"count": len(docs), "ids": self.doc_ids},
            )
            docs = [d.to_dict() for d in docs]
        else: 
            docs = self.memory.ollama_embeddings.search_related_documents(
                goal_text, self.top_k
            )
            self.logger.log(
                "DocumentsSearched",
                {"count": len(docs), "goal_text": goal_text, "top_k": self.top_k},
            )

        context[self.output_key] = docs
        context["retrieved_ids"] = [d["id"] for d in docs]
        self.logger.log(
            "KnowledgeDBLoaded",
            {"count": len(docs), "search_method": self.search_method},
        )

        return context
