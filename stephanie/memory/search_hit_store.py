# stephanie/memory/search_hit_store.py
from __future__ import annotations

from typing import Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.search_hit import SearchHitORM


class SearchHitStore(BaseSQLAlchemyStore):
    orm_model = SearchHitORM
    default_order_by = SearchHitORM.created_at

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "search_hits"

    def add_hit(self, hit_data: dict) -> SearchHitORM:
        """Insert a single search hit."""
        def op(s):
            hit = SearchHitORM(**hit_data)
            s.add(hit)
            s.flush()
            s.refresh(hit)
            return hit

        return self._run(op)

    def bulk_add_hits(self, hit_data_list: list[dict]) -> list[SearchHitORM]:
        """Insert multiple search hits at once."""
        def op(s):
            hits = [SearchHitORM(**data) for data in hit_data_list]
            s.add_all(hits)
            s.flush()
            return hits

        return self._run(op)    

    def get_by_id(self, hit_id: int) -> Optional[SearchHitORM]:
        """Fetch a search hit by its ID."""
        def op(s):
            return s.query(SearchHitORM).filter_by(id=hit_id).first()

        return self._run(op)

    def get_all_for_goal(self, goal_id: int) -> list[SearchHitORM]:
        """Fetch all hits for a given goal ID."""
        def op(s):
            return s.query(SearchHitORM).filter_by(goal_id=goal_id).all()

        return self._run(op)

    def delete_by_id(self, hit_id: int) -> bool:
        """Delete a search hit by ID."""
        def op(s):
            hit = s.query(SearchHitORM).filter_by(id=hit_id).first()
            if hit:
                s.delete(hit)
                return True
            return False

        return self._run(op)
