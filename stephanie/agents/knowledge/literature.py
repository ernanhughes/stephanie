# stephanie/agents/knowledge/literature.py

import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.tools import WebSearchTool
from stephanie.utils.file_utils import write_text_to_file


class LiteratureAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.strategy = cfg.get("strategy", "query_and_summarize")
        self.preferences = cfg.get("preferences", ["goal_consistency", "novelty"])
        self.max_results = cfg.get("max_results", 5)
        self.web_search_tool = WebSearchTool(cfg.get("web_search", {}), self.logger)

        self.logger.log(
            "LiteratureAgentInit",
            {
                "strategy": self.strategy,
                "preferences": self.preferences,
                "max_results": self.max_results,
            },
        )

    async def run(self, context: dict) -> dict:
        self.logger.log("LiteratureQuery", {"context": context})
        goal = self.extract_goal_text(context.get(GOAL))

        # Step 1: Generate search query using LLM
        search_query = self._generate_search_query(context)
        if not search_query:
            self.logger.log("LiteratureQueryFailed", {"goal": goal})
            return context

        self.logger.log("SearchingWeb", {"query": search_query, "goal": goal})

        # Step 2: Perform web search
        results = await self.web_search_tool.search(
            search_query, max_results=self.max_results
        )

        if not results:
            self.logger.log(
                "NoResultsFromWebSearch",
                {
                    "goal_snippet": goal[:60],
                    "search_query": search_query,
                },
            )
            return context

        self.logger.log("SearchResult", {"results": results})

        # Step 3: Parse each result with LLM
        parsed_results = []
        for result in results:
            summary_context = {
                **{
                    "title": result.get("title", "no Title"),
                    "link": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                    "page": result.get("page", ""),
                },
                **context,
            }

            summary = self._summarize_result(summary_context)

            if summary.strip():
                parsed_results.append(f"""
                    [Title: {result["title"]}]({result["url"]})\n
                    Summary: {summary}
                """)

        self.logger.log(
            "LiteratureSearchCompleted",
            {
                "total_results": len(parsed_results),
                "goal": goal,
                "search_query": search_query,
            },
        )

        context["literature"] = parsed_results

        return context

    def _generate_search_query(self, context: dict) -> str:
        try:
            prompt = self.prompt_loader.load_prompt(self.cfg, context)
            self.logger.log(
                "LLMPromptGenerated_SearchQuery", {"prompt_snippet": prompt[:200]}
            )

            response = self.call_llm(prompt, context)
            self.logger.log(
                "LLMResponseReceived_SearchQuery", {"response_snippet": response[:200]}
            )

            # Structured format
            match = re.search(r"search query:<([^>]+)>", response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

            # Fallback format
            match = re.search(
                r"(?:query|search)[:\s]+\"([^\"]+)\"", response, re.IGNORECASE
            )
            if match:
                query = match.group(1).strip()
                self.logger.log("SearchQuery", {"Search Query": query})
                return query

            # Fallback to goal
            goal = self.extract_goal_text(context.get(GOAL))
            self.logger.log("FallingBackToGoalAsQuery", {"goal": goal})
            return f"{goal} productivity study"

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log("LiteratureQueryGenerationFailed", {"error": str(e)})
            return f"{context.get('goal', '')} remote work meta-analysis"

    def _summarize_result(self, context: dict) -> str:
        try:
            prompt = self.prompt_loader.from_file(
                self.cfg.get("parse_prompt", "parse.txt"), self.cfg, context
            )
            self.logger.log(
                "LLMPromptGenerated_Summarize",
                {"title": context.get("title", ""), "prompt_snippet": prompt[:200]},
            )

            raw_summary = self.call_llm(prompt, context)
            self.logger.log(
                "LLMResponseReceived_Summarize",
                {
                    "title": context.get("title", ""),
                    "response_snippet": raw_summary[:200],
                },
            )

            # Try extracting "Summary" section
            summary_match = re.search(
                r"Summary\s*\n(?:.*\n)*?\s*(.+?)(?=\n#|\Z)",
                raw_summary,
                re.DOTALL | re.IGNORECASE,
            )
            if summary_match:
                return summary_match.group(1).strip()

            # Fallback: first paragraph of sufficient length
            lines = raw_summary.splitlines()
            for line in lines:
                if len(line.strip()) > 50:
                    return line.strip()

            return ""

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log("FailedToParseLiterature", {"error": str(e)})
            return ""
