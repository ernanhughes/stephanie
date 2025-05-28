# co_ai/agents/knowledge_loader.py
from co_ai.agents.base import BaseAgent
from co_ai.models import SearchResultORM

class KnowledgeLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get("goal")
        goal_id = goal.get("id")

        # Try to load from DB
        knowledge_base = self._load_from_db(goal_id)

        if knowledge_base:
            self.logger.log("LoadedFromDB", {"count": len(knowledge_base)})
            context[self.output_key] = knowledge_base
        else:
            self.logger.log("NoResultsFoundInDB", {})

        return context

    def _load_from_db(self, goal_id: int) -> list:
        """
        Load processed search results (with key concepts, insights, etc.)
        that are linked to this goal.
        """

        results = self.memory.search_results.get_by_goal_id(goal_id)
        return [
            {
                "title": r.title,
                "summary": r.summary,
                "refined_summary": r.refined_summary,
                "url": r.url,
                "source": r.source,
                "key_concepts": r.key_concepts,
                "technical_insights": r.technical_insights,
                "relevance_score": r.relevance_score,
                "related_ideas": r.related_ideas,
                "extracted_methods": r.extracted_methods,
                "domain_knowledge_tags": r.domain_knowledge_tags
            }
            for r in results
        ]