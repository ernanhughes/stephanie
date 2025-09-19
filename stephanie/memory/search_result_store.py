# stephanie/memory/search_result_store.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.search_result import SearchResultORM


class SearchResultStore(BaseSQLAlchemyStore):
    orm_model = SearchResultORM
    default_order_by = SearchResultORM.created_at

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "search_results"

    def name(self) -> str:
        return self.name

    def add_result(
        self,
        *,
        query: str,
        source: str,
        result_type: str,
        title: str,
        summary: str,
        url: str,
        author: Optional[str] = None,
        published_at: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        goal_id: Optional[int] = None,
        parent_goal: Optional[str] = None,
        strategy: Optional[str] = None,
        focus_area: Optional[str] = None,
        extra_data: Optional[Dict] = None,
    ) -> SearchResultORM:
        """Insert a single search result."""
        def op(s):
            result = SearchResultORM(
                query=query,
                source=source,
                result_type=result_type,
                title=title,
                summary=summary,
                url=url,
                author=author,
                published_at=published_at,
                tags=tags,
                goal_id=goal_id,
                parent_goal=parent_goal,
                strategy=strategy,
                focus_area=focus_area,
                extra_data=extra_data,
            )
            s.add(result)
            s.flush()
            s.refresh(result)
            return result

        return self._run(op, commit=True)

    def bulk_add_results(self, results: List[Dict]) -> List[SearchResultORM]:
        """Insert multiple search results at once."""
        def op(s):
            orm_objects = [SearchResultORM(**result) for result in results]
            s.add_all(orm_objects)
            s.flush()
            return orm_objects

        return self._run(op, commit=True)

    def get_by_goal_id(self, goal_id: int) -> List[SearchResultORM]:
        """Retrieve all search results for a given goal."""
        def op(s):
            return s.query(SearchResultORM).filter_by(goal_id=goal_id).all()

        return self._run(op)

    def get_by_strategy_and_focus(
        self, strategy: str, focus_area: str
    ) -> List[SearchResultORM]:
        """Retrieve results filtered by strategy and focus area."""
        def op(s):
            return (
                s.query(SearchResultORM)
                .filter(
                    SearchResultORM.strategy == strategy,
                    SearchResultORM.focus_area == focus_area,
                )
                .all()
            )

        return self._run(op)

    def get_by_source_and_type(
        self, source: str, result_type: str
    ) -> List[SearchResultORM]:
        """Retrieve results filtered by source and type."""
        def op(s):
            return (
                s.query(SearchResultORM)
                .filter(
                    SearchResultORM.source == source,
                    SearchResultORM.result_type == result_type,
                )
                .all()
            )

        return self._run(op)

    def delete_by_goal_id(self, goal_id: int) -> None:
        """Delete all results linked to a goal."""
        def op(s):
            s.query(SearchResultORM).filter_by(goal_id=goal_id).delete()

        self._run(op, commit=True)

    def clear_all(self) -> None:
        """Delete all results (useful for testing)."""
        def op(s):
            s.query(SearchResultORM).delete()

        self._run(op, commit=True)
