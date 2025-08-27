# stephanie/agents/knowledge/arxiv_search.py
import re
from datetime import datetime, timedelta
import arxiv

from stephanie.agents.base_agent import BaseAgent


class ArxivSearchAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.year_start = cfg.get("year_start", 2021)
        self.year_end = cfg.get("year_end", 2025)
        self.category = cfg.get("category", "cs.AI")
        self.max_results = cfg.get("max_results", 50)
        self.return_top_n = cfg.get("top_n", 10)
        self.date_filter = cfg.get("date_filter", "")

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {}).get("goal_text", "")

        # --- Report: Start ---
        self.report({
            "event": "start",
            "step": "ArxivSearch",
            "details": f"Searching arXiv for goal: {goal}",
        })

        # Step 1: Extract relevant keywords
        keywords = self.extract_keywords(context)
        context["search_keywords"] = keywords

        self.report({
            "event": "keywords_extracted",
            "step": "ArxivSearch",
            "details": f"Extracted {len(keywords)} keywords",
            "keywords": keywords,
        })

        # Step 2: Build Arxiv-compatible query
        query = self.build_arxiv_query_from_goal(
            context=context,
            year_start=self.year_start,
            year_end=self.year_end,
            category=self.category,
            keywords=keywords,
        )

        self.report({
            "event": "query_built",
            "step": "ArxivSearch",
            "details": f"Built query with {len(keywords)} keywords",
            "query": query,
        })

        # Step 3: Fetch raw papers
        results = []
        try:
            results = self.fetch_arxiv_results(
                context, query, max_results=self.max_results
            )
            context["raw_arxiv_results"] = results

            self.report({
                "event": "search_complete",
                "step": "ArxivSearch",
                "details": f"Fetched {len(results)} papers",
                "sample_titles": [r['title'] for r in results[:3]],  # just first 3
            })
        except Exception as e:
            self.report({
                "event": "error",
                "step": "ArxivSearch",
                "details": f"Error fetching arXiv results: {str(e)}",
            })

        # TODO: Ranking can also report
        # context["filtered_arxiv_results"] = top_ranked

        context[self.output_key] = results

        # --- Report: End ---
        self.report({
            "event": "end",
            "step": "ArxivSearch",
            "details": f"Completed with {len(results)} results",
        })

        return context

    def extract_keywords(self, merged_context: dict) -> list:
        """Extract keywords from the goal text using simple heuristics."""
        response = self.execute_prompt(merged_context)
        pattern = r"(?:\n|\r|\r\n)([^\n\r]+?)(?=(?:\n|\r|\r\n|$))"
        lines = re.findall(pattern, response.strip())
        keywords = [re.sub(r"^[-•\d\.\s]+", "", line).strip() for line in lines]

        # Keep the debug log
        self.logger.log(
            "KeywordsExtracted", {"raw_keywords": lines, "cleaned_keywords": keywords}
        )
        return [kw for kw in keywords if kw]

    def build_arxiv_query_from_goal(
        self,
        context: dict,
        keywords: list[str],
        year_start: int = None,
        year_end: int = None,
        category: str = None,
    ) -> str:
        keyword_filter = " OR ".join(f'"{kw.strip()}"' for kw in keywords if kw.strip())
        filters = [f"({keyword_filter})"]

        date_filter_mode = self.cfg.get("date_filter", "").lower()
        now = datetime.now()

        if date_filter_mode == "today":
            day = now.strftime("%Y%m%d")
            filters.append(f"submittedDate:[{day} TO {day}]")
        elif date_filter_mode == "week":
            start = (now - timedelta(days=7)).strftime("%Y%m%d")
            end = now.strftime("%Y%m%d")
            filters.append(f"submittedDate:[{start} TO {end}]")
        elif date_filter_mode == "month":
            start = (now - timedelta(days=30)).strftime("%Y%m%d")
            end = now.strftime("%Y%m%d")
            filters.append(f"submittedDate:[{start} TO {end}]")
        elif date_filter_mode == "year":
            start = (now - timedelta(days=365)).strftime("%Y%m%d")
            end = now.strftime("%Y%m%d")
            filters.append(f"submittedDate:[{start} TO {end}]")
        elif year_start or year_end:
            start = f"{year_start}0101" if year_start else "00000101"
            end = f"{year_end}1231" if year_end else "99991231"
            filters.append(f"submittedDate:[{start} TO {end}]")

        if category:
            filters.append(f"cat:{category}")

        return " AND ".join(filters)

    def fetch_arxiv_results(
        self, context: dict, query: str, max_results: int = 50
    ) -> list[dict]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )

        results = []
        goal = context.get("goal", {})
        goal_id = goal.get("id", "")
        parent_goal = goal.get("goal_text")
        strategy = goal.get("strategy")
        focus_area = goal.get("focus_area")

        for result in search.results():
            arxiv_url = result.entry_id
            pid = arxiv_url.split("/")[-1]
            results.append(
                {
                    "query": query,
                    "source": self.name,
                    "result_type": "paper",
                    "title": result.title.strip(),
                    "summary": result.summary.strip(),
                    "url": f"https://arxiv.org/pdf/{pid}.pdf",
                    "goal_id": goal_id,
                    "parent_goal": parent_goal,
                    "strategy": strategy,
                    "focus_area": focus_area,
                    "authors": [a.name for a in result.authors],
                    "published": result.published.isoformat(),
                    "pid": pid,
                    "primary_category": result.primary_category,
                }
            )

        return results
