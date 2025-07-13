# stephanie/agents/knowledge/__init__.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class KnowledgeDBLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.top_k = cfg.get("top_k", 10)
        self.include_full_text = cfg.get("include_full_text", False)
        self.search_method = cfg.get("search_method", "document")  # or "section"

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get("goal_text", "")

        if not goal_text:
            self.logger.log("KnowledgeDBLoadSkipped", {"reason": "Missing goal"})
            return context

        # Step 1: Get the goal embedding
        goal_embedding = self.memory.embedding.get_or_create(goal_text)

        # Step 2: Search in the database for similar documents or sections
        if self.search_method == "section":
            matches = self.memory.embedding.search_similar_documents_with_scores(
                goal_text, top_k=self.top_k
            )
        else:
            matches = self.memory.embedding.search_related(goal_text, top_k=self.top_k)

        # Step 3: Format results
        results = []
        for match in matches:
            entry = {
                "id": match.get("id"),
                "title": match.get("title", match.get("prompt", "Unknown")),
                "score": match.get("score", 0.0),
                "content": match.get("text", match.get("prompt", "")),
                "source": match.get("source", self.search_method),
            }
            results.append(entry)

        context[self.output_key] = results
        context["retrieved_ids"] = [r["id"] for r in results if r.get("id")]
        self.logger.log(
            "KnowledgeDBLoaded",
            {"count": len(results), "search_method": self.search_method},
        )

        return context
