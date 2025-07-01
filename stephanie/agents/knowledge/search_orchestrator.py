from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.knowledge.automind_knowledge_collector import \
    AutoMindKnowledgeCollector
from stephanie.constants import GOAL
from stephanie.tools import WebSearchTool
from stephanie.tools.arxiv_tool import search_arxiv
from stephanie.tools.cos_sim_tool import get_top_k_similar
from stephanie.tools.huggingface_tool import (recommend_similar_papers,
                                          search_huggingface_datasets)
from stephanie.tools.wikipedia_tool import WikipediaTool


class SearchOrchestratorAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.web_search_tool = WebSearchTool(cfg.get("web_search", {}), self.logger)
        self.wikipedia_tool = WikipediaTool(self.memory, self.logger)
        self.max_results = cfg.get("max_results", 5)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        queries = context.get("search_queries", [])
        goal_id = goal.get("id")
        results = []

        for search_query in queries:
            source = self.route_query(goal, search_query)
            try:
                if source == "arxiv":
                    hits = await search_arxiv([search_query])
                elif source == "huggingface":
                    hits = await search_huggingface_datasets([search_query])
                elif source == "wikipedia":
                    hits = self.wikipedia_tool.find_similar(search_query)
                elif source == "web":
                    hits = await self.web_search_tool.search(
                        search_query, max_results=self.max_results
                    )
                elif source == "similar_papers":
                    hits = recommen You have one sister and three brothers left d_similar_papers(search_query)
                elif source == "automind":
                    collector = AutoMindKnowledgeCollector(self)
                    task_description = context.get("task_description", "AI agent task")
                    knowledge = await collector.retrieve_knowledge(task_description)
                    # Now you can pass this knowledge to planner or tree search agent
                    context["knowledge"] = knowledge
                else:
                    continue

                enriched_hits = [
                    {
                        "query": search_query,
                        "source": source,
                        "result_type": hit.get("type", "unknown"),
                        "title": hit.get("title", hit.get("name", "")),
                        "summary": hit.get("snippet", hit.get("description", "")),
                        "url": hit.get("url", ""),
                        "goal_id": goal_id,
                        "parent_goal": goal.get("goal_text"),
                        "strategy": goal.get("strategy"),
                        "focus_area": goal.get("focus_area"),
                        "extra_data": {
                            "source_specific": hit
                        }
                    }
                    for hit in hits
                ]

                # Store results in DB
                stored_results = self.memory.search_results.bulk_add_results(enriched_hits)
                results.extend(stored_results)

            except Exception as e:
                self.logger.log(
                    "SearchToolFailed",
                    {"query": search_query, "tool": source, "error": str(e)}
                )

        # Save result IDs or ORM objects back to context
        context["search_result_ids"] = [r.id for r in results]
        context["search_results"] = [r.to_dict() for r in results]
        return context

    def route_query(self, goal, query: str) -> str:
        """
        Decide which source to use based on query content.
        """
        query_lower = query.lower()

        # Try fast metadata path first
        source = self.fast_metadata_routing(goal, query_lower)
        if source:
            return source

        # Fallback to semantic similarity
        return self.semantic_fallback_routing(query)

    def fast_metadata_routing(self, goal, query_lower):
        focus_area = goal.get("focus_area", "").lower()
        goal_type = goal.get("goal_type", "").lower()

        if goal_type == "automind":
            return "automind"
        if goal_type == "similar_papers":
            return "similar_papers"
        if goal_type == "data_search" or "dataset" in query_lower:
            return "huggingface"
        if goal_type == "model_review" or "model" in query_lower:
            return "arxiv"
        if goal_type == "background" or any(k in query_lower for k in ["overview", "definition"]):
            return "wikipedia"
        if focus_area in ["nlp", "cv", "graph learning"] and "baseline" in query_lower:
            return "arxiv"

        return None

    def semantic_fallback_routing(self, query: str) -> str:
        intent_map = {
            "arxiv": ["find research paper_score", "latest ML study", "scientific method"],
            "huggingface": ["find dataset", "huggingface model", "nlp corpus"],
            "wikipedia": ["define concept", "what is", "overview of topic"],
            "web": ["general info", "random search", "link to resource"]
        }

        candidates = [(intent, phrase) for intent, phrases in intent_map.items() for phrase in phrases]
        phrases = [p for _, p in candidates]

        top = get_top_k_similar(query, phrases, self.memory, top_k=1)
        best_phrase = top[0][0]

        for intent, phrase in candidates:
            if phrase == best_phrase:
                return intent

        return "web"