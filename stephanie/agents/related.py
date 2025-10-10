# stephanie/agents/search/goal_document_search.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class RelatedAgent(BaseAgent):
    """
    Agent that searches for documents related to a given goal using embeddings.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.document_type = cfg.get("document_type", "document")
        self.embedding_type = self.memory.embedding.name  # e.g., "hnet"
        self.top_k = cfg.get("top_k", 10)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get("goal_text")

        # Step 1: Search for related documents
        results = self.memory.embedding.search_related_scorables(
            query=goal_text,
            top_k=self.top_k,
            target_type=self.document_type,
        )

        # Step 2: Store results in context
        context[self.output_key] = results

        # Step 3: Report search outcome
        self.report({
            "event": "goal_document_search",
            "document_type": self.document_type,
            "embedding_type": self.embedding_type,
            "top_k": self.top_k,
            "returned": len(results),
            "examples": [
                {
                    "id": r.get("id"),
                    "title": r.get("title"),
                    "score": r.get("score"),
                }
                for r in results[:3]  # only include top 3 in the report for readability
            ],
        })

        return context
