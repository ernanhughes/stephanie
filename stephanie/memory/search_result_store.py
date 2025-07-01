# stores/search_result_store.py
from datetime import datetime
from typing import Dict, List, Optional

from stephanie.memory import BaseStore
from stephanie.models.search_result import SearchResultORM


class SearchResultStore(BaseStore):
    def __init__(self, session, logger):
        super().__init__(session, logger)
        self.name = "search_results"

    def name(self) -> str:
        return "search_results"

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
        extra_data: Optional[Dict] = None
    ) -> SearchResultORM:
        """
        Add a single search result to the database.
        """
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
            extra_data=extra_data
        )
        self.session.add(result)
        self.session.commit()
        self.session.refresh(result)
        return result

    def bulk_add_results(self, results: List[Dict]) -> List[SearchResultORM]:
        """
        Add multiple search results at once.
        Each dict should contain the required fields.
        """
        orm_objects = [SearchResultORM(**result) for result in results]
        self.db.bulk_save_objects(orm_objects)
        self.db.commit()
        return orm_objects

    def get_by_goal_id(self, goal_id: int) -> List[SearchResultORM]:
        """
        Retrieve all search results associated with a specific goal.
        """
        return self.db.query(SearchResultORM).filter(
            SearchResultORM.goal_id == goal_id
        ).all()

    def get_by_strategy_and_focus(self, strategy: str, focus_area: str) -> List[SearchResultORM]:
        """
        Get results filtered by strategy and focus area.
        """
        return self.db.query(SearchResultORM).filter(
            SearchResultORM.strategy == strategy,
            SearchResultORM.focus_area == focus_area
        ).all()

    def get_by_source_and_type(self, source: str, result_type: str) -> List[SearchResultORM]:
        """
        Get results filtered by source and type (e.g., arxiv/paper_score).
        """
        return self.db.query(SearchResultORM).filter(
            SearchResultORM.source == source,
            SearchResultORM.result_type == result_type
        ).all()

    def delete_by_goal_id(self, goal_id: int) -> None:
        """
        Delete all search results linked to a given goal.
        """
        self.db.query(SearchResultORM).filter(
            SearchResultORM.goal_id == goal_id
        ).delete()
        self.db.commit()

    def clear_all(self) -> None:
        """
        Delete all records — useful for testing.
        """
        self.db.query(SearchResultORM).delete()
        self.db.commit()