# stephanie/agents/knowledge/arxiv_search.py
"""
arXiv Search Agent Module

Provides functionality to search and retrieve academic papers from arXiv.org
based on research goals and extracted keywords. Includes robust error handling,
query construction, and result processing capabilities.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import arxiv

from stephanie.agents.base_agent import BaseAgent


class ArxivSearchAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # Configuration with defaults
        self.year_start = cfg.get("year_start", 2021)
        self.year_end = cfg.get("year_end", 2025)
        self.category = cfg.get("category", "cs.AI")  # Default to AI category
        self.max_results = cfg.get("max_results", 50)
        self.return_top_n = cfg.get("top_n", 10)
        self.date_filter = cfg.get("date_filter", "")

    async def run(self, context: dict) -> dict:
        """Main execution method for arXiv search agent"""
        goal = context.get("goal", {}).get("goal_text", "")

        # --- Performance reporting ---
        self.report({
            "event": "start",
            "step": "ArxivSearch",
            "details": f"Searching arXiv for goal: {goal}",
        })

        # Step 1: Keyword extraction
        keywords = self.extract_keywords(context)
        context["search_keywords"] = keywords

        self.report({
            "event": "keywords_extracted",
            "step": "ArxivSearch",
            "details": f"Extracted {len(keywords)} keywords",
            "keywords": keywords,
        })

        # Step 2: Query construction
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

        # Step 3: Fetch results
        results = []
        try:
            results = self.fetch_arxiv_results(
                context, query, max_results=self.max_results
            )
            context["raw_arxiv_results"] = results

            len_results = len(results) if results else 0

            self.report({
                "event": "search_complete",
                "step": "ArxivSearch",
                "details": f"Fetched {len_results} papers",
                "sample_titles": [r['title'] for r in results[:3]],  # First 3 titles
            })
        except Exception as e:
            self.report({
                "event": "error",
                "step": "ArxivSearch",
                "details": f"Error fetching arXiv results: {str(e)}",
            })

        context[self.output_key] = results

        # --- Completion report ---
        self.report({
            "event": "end",
            "step": "ArxivSearch",
            "details": f"Completed with {len(results)} results",
        })

        return context

    def extract_keywords(self, merged_context: dict) -> list:
        """Extract keywords using prompt-based approach with regex parsing"""
        response = self.execute_prompt(merged_context)
        # Match lines separated by newlines
        pattern = r"(?:\n|\r|\r\n)([^\n\r]+)(?=(?:\n|\r|\r\n|$))"
        lines = re.findall(pattern, response.strip())
        # Clean numbering/bullets from lines
        keywords = [re.sub(r"^[-•\d\.\s]+", "", line).strip() for line in lines]

        # Debug logging
        self.logger.log(
            "KeywordsExtracted", {"raw_keywords": lines, "cleaned_keywords": keywords}
        )
        return [kw for kw in keywords if kw]  # Return non-empty keywords

    def build_arxiv_query_from_goal(
        self,
        context: dict,
        keywords: list[str],
        year_start: int = None,
        year_end: int = None,
        category: str = None,
    ) -> str:
        """Construct arXiv-compatible search query with filters"""
        keyword_filter = " OR ".join(f'"{kw.strip()}"' for kw in keywords if kw.strip())
        filters = [f"({keyword_filter})"]

        # Date filtering logic
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
            # Use year range if specified
            start = f"{year_start}0101" if year_start else "00000101"
            end = f"{year_end}1231" if year_end else "99991231"
            filters.append(f"submittedDate:[{start} TO {end}]")

        # Category filter
        if category:
            filters.append(f"cat:{category}")

        return " AND ".join(filters)

    def fetch_arxiv_results(
        self, context: dict, query: str, max_results: int = 50
    ) -> list[dict]:
        """
        Fetch papers from arXiv API with error handling and retry logic.
        Returns list of paper dictionaries with metadata.
        """
        results: list[dict] = []
        goal = context.get("goal", {})
        goal_id = goal.get("id", "")
        parent_goal = goal.get("goal_text")
        strategy = goal.get("strategy")
        focus_area = goal.get("focus_area")

        # Safety clamp on results
        max_results = min(max_results or self.max_results, 100)

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )

            # Retry mechanism for transient errors
            for attempt in range(3):
                try:
                    for result in search.results():
                        try:
                            arxiv_url = getattr(result, "entry_id", "") or ""
                            pid = arxiv_url.split("/")[-1] if "/" in arxiv_url else None
                            if not pid:
                                continue  # Skip entries without ID

                            # Extract and clean fields
                            title = (getattr(result, "title", "") or "Unknown").strip()
                            summary = (getattr(result, "summary", "") or "").strip()
                            published = getattr(result, "published", None)
                            published_str = (
                                published.isoformat() if published else ""
                            )
                            authors = [
                                a.name for a in getattr(result, "authors", []) or []
                            ]
                            primary_category = getattr(
                                result, "primary_category", "unknown"
                            )

                            # Build result dictionary
                            results.append(
                                {
                                    "query": query,
                                    "source": self.name,
                                    "result_type": "paper",
                                    "title": title,
                                    "summary": summary,
                                    "url": f"https://arxiv.org/pdf/{pid}.pdf",
                                    "goal_id": goal_id,
                                    "parent_goal": parent_goal,
                                    "strategy": strategy,
                                    "focus_area": focus_area,
                                    "authors": authors,
                                    "published": published_str,
                                    "pid": pid,
                                    "primary_category": primary_category,
                                }
                            )
                        except Exception as parse_err:
                            self.logger.log(
                                "ArxivResultParseError",
                                {
                                    "error": str(parse_err),
                                    "entry_id": getattr(result, "entry_id", "unknown"),
                                },
                            )
                            continue

                    break  # Success - exit retry loop

                except arxiv.UnexpectedEmptyPageError as e:
                    self.logger.log(
                        "ArxivEmptyPageRetry",
                        {"query": query, "attempt": attempt + 1, "error": str(e)},
                    )
                    if attempt < 2:
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return []  # Failed after retries

        except Exception as e:
            self.logger.log(
                "ArxivSearchFailed",
                {"query": query, "error": str(e)},
            )
            return []

        # Log results
        if not results:
            self.logger.log(
                "ArxivNoResults",
                {"query": query, "goal_id": goal_id, "parent_goal": parent_goal},
            )
        else:
            self.logger.log(
                "ArxivResultsFetched",
                {
                    "query": query, 
                    "count": len(results), 
                    "first_title": results[0]["title"]
                },
            )

        return results