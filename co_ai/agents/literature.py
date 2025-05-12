# co_ai/agents/literature.py
import re

from co_ai.agents.base import BaseAgent
from co_ai.logs import JSONLogger
from co_ai.memory import VectorMemory
from co_ai.tools import WebSearchTool


class LiteratureAgent(BaseAgent):
    """
    The LiteratureAgent turns a research goal into a web search query,
    retrieves recent research findings, and parses them into usable summaries.
    
    From the paper:
    > 'The Generation agent iteratively searches the web, retrieves and reads relevant research articles'
    """

    def __init__(self, cfg, memory:VectorMemory=None, logger:JSONLogger=None):
        super().__init__(cfg, memory, logger)
        self.strategy = cfg.get("strategy", "query_and_summarize")
        self.preferences = cfg.get("preferences", ["goal_consistency", "novelty"])
        self.max_results = cfg.get("max_results", 5)

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
        if self.cfg.get("skip_if_completed", False):
            results  = self._get_completed(context)
            if results:
                self.logger.log("LiteratureSearchSkipped", {
                    "goal": context.get("goal", "")
                })
                return results

        self.log(f"Searching literature for: {context}")

        # Step 1: Generate search query using LLM
        search_query = self._generate_search_query(context)
        goal = context.get("goal", "")
        if not search_query:
            self.logger.log("LiteratureQueryFailed", {"goal": goal})
            return context

        self.log("SearchingWeb", {"query": search_query, "goal": goal})

        # Step 2: Perform web search
        results = self.web_search_tool.search(search_query, max_results=self.max_results)
        if not results:
            self.logger.log("NoResultsFromWebSearch", {
                "goal_snippet": goal[:60],
                "search_query": search_query
            })
            return context
        self.log("SearchResult", {"results": results})

        # Step 3: Parse each result with LLM
        parsed_results = []
        for result in results:
            summary_context = {**{"title": result["title"], "link":result["href"], "snippet":result["body"]}, **context}
            summary = self._summarize_result(summary_context)
            if summary.strip():
                parsed_results.append(f"""
                    [Title: {result["title"]}]({result["href"]})\n
                    Summary: {summary}
                    """
                )

        # Log full search results
        self.logger.log("LiteratureSearchCompleted", {
            "total_results": len(parsed_results),
            "goal": goal,
            "search_query": search_query
        })

        # Store in context
        context["literature"] = parsed_results
        # context["literature_data"] = results

        self._save_context(context)
        return context

    def _generate_search_query(self, context: dict) -> str:
        """Use LLM to turn natural language goal into a precise search query."""
        try:
            prompt = self.prompt_loader.load_prompt(self.cfg, context)
            response = self.call_llm(prompt)

            # Try matching structured format first
            match = re.search(r"search query:<([^>]+)>", response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

            # Fallback: use keyword-based parsing
            match = re.search(r"(?:query|search)[:\s]+\"([^\"]+)\"", response, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                self.logger.log("SearchQuery",{"Search Query": query})
                return query

            # Final fallback: use goal as-is
            goal =context.get("goal", "")
            self.logger.log("FallingBackToGoalAsQuery", {"goal": goal})
            return f"{goal} productivity study"

        except Exception as e:
            self.logger.log("LiteratureQueryGenerationFailed", {"error": str(e)})
            return f'{context.get("goal", "")} remote work meta-analysis'

    def _summarize_result(self, context:dict) -> str:
        """Ask LLM to extract key insights from article metadata."""
        try:
            prompt = self.prompt_loader.from_file(self.cfg.parse_prompt, self.cfg, context)
            raw_summary = self.call_llm(prompt).strip()

            # Extract summary section if present
            summary_match = re.search(
                r"Summary\s*\n(?:.*\n)*?\s*(.+?)(?=\n#|\Z)",
                raw_summary,
                re.DOTALL | re.IGNORECASE
            )
            if summary_match:
                return summary_match.group(1).strip()

            # Fallback: extract any paragraph
            lines = raw_summary.splitlines()
            for line in lines:
                if len(line.strip()) > 50:
                    return line.strip()

            return ""

        except Exception as e:
            self.logger.log("FailedToParseLiterature", {"error": str(e)})
            return ""