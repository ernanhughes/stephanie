# co_ai/agents/literature.py
import re
from typing import Dict, Any, List

from co_ai.agents.base import BaseAgent
from co_ai.tools import WebSearchTool
from co_ai.utils import load_prompt_from_file


class LiteratureAgent(BaseAgent):
    """
    The LiteratureAgent turns a research goal into a web search query,
    retrieves recent research findings, and parses them into usable summaries.
    
    From the paper:
    > 'The Generation agent iteratively searches the web, retrieves and reads relevant research articles'
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.strategy = cfg.get("strategy", "query_and_summarize")
        self.preferences = cfg.get("preferences", ["goal_consistency", "novelty"])
        self.max_results = cfg.get("max_results", 5)

        # Load prompts
        self.query_prompt = load_prompt_from_file("literature_query.txt")
        self.parse_prompt = load_prompt_from_file("literature_parse.txt")
        self.web_search_tool = WebSearchTool()
        self.logger.log("LiteratureAgentInit", {
            "strategy": self.strategy,
            "preferences": self.preferences,
            "max_results": self.max_results
        })

    async def run(self, context: dict) -> dict:
        """
        Run literature search based on current goal and preferences.
        
        Args:
            context: Dictionary with keys:
                - goal: Research objective
                - preferences: Optional override of evaluation criteria
        """
        goal = context.get("goal", "Unknown goal")
        self.log(f"Searching literature for: {goal}")

        # Step 1: Generate search query using LLM
        search_query = self._generate_search_query(goal)
        if not search_query:
            self.logger.log("LiteratureQueryFailed", {"goal": goal})
            return context

        self.log("SearchingWeb", {"query": search_query, "goal": goal})
        # Step 2: Perform web search
        results = self.web_search_tool.search(search_query, max_results=self.max_results)
        self.log("SearchResult", {"results": results})
        if not results:
            self.logger.log("NoResultsFromWebSearch", {
                "goal_snippet": goal[:60],
                "search_query": search_query
            })
            return context

        # Step 3: Parse each result with LLM
        parsed_results = []
        for result in results:
            summary = self._summarize_result(result["title"], result["href"], result["body"])
            if summary.strip():
                parsed_results.append({
                    "title": result["title"],
                    "href": result["href"],
                    "summary": summary
                })

        # Log full search results
        self.logger.log("LiteratureSearchCompleted", {
            "total_results": len(parsed_results),
            "goal": goal,
            "search_query": search_query
        })

        # Store in context
        context["literature"] = parsed_results
        return context

    def _generate_search_query(self, goal: str) -> str:
        """Use LLM to turn natural language goal into a precise search query."""
        try:
            response = self.call_llm(self.query_prompt.format(goal=goal))

            # Try matching structured format first
            match = re.search(r"search query:<([^>]+)>", response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

            # Fallback: use keyword-based parsing
            match = re.search(r"(?:query|search)[:\s]+\"([^\"]+)\"", response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

            # Final fallback: use goal as-is
            self.logger.log("FallingBackToGoalAsQuery", {"goal": goal})
            return f"{goal} productivity study"

        except Exception as e:
            self.logger.log("LiteratureQueryGenerationFailed", {"error": str(e)})
            return f"{goal} remote work meta-analysis"

    def _summarize_result(self, title: str, link: str, snippet: str) -> str:
        """Ask LLM to extract key insights from article metadata."""
        try:
            prompt = self.parse_prompt.format(title=title, link=link, snippet=snippet)
            raw_summary = self.call_llm(prompt).strip()

            # Extract summary section if present
            summary_match = re.search(r"# Summary\n(.+?)(?=\n#|\Z)", raw_summary, re.DOTALL)
            if summary_match:
                return summary_match.group(1).strip()

            # Fallback: extract any paragraph
            lines = raw_summary.splitlines()
            for line in lines:
                if len(line.strip()) > 50:
                    return line.strip()

            return ""

        except Exception as e:
            self.logger.log("FailedToParseLiterature", {"error": str(e), "title": title})
            return ""