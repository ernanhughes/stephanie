# stephanie/memory/scorable_rank_store.py
from __future__ import annotations

from typing import List

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.scorable_rank import ScorableRankORM


class ScorableRankStore(BaseSQLAlchemyStore):
    orm_model = ScorableRankORM
    default_order_by = ScorableRankORM.created_at.asc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "scorable_ranks"

    def insert(self, data: dict) -> int:
        """Insert a single rank record."""
        def op(s):
            obj = ScorableRankORM(**data)
            
            s.add(obj)
            if self.logger:
                self.logger.log("ScorableRankInserted", obj.to_dict())
            return obj.id
        return self._run(op)

    def bulk_insert(self, data_list: List[dict]) -> List[int]:
        """Insert multiple rank records at once."""
        def op(s):
            objs = [ScorableRankORM(**d) for d in data_list]
            
            s.add_all(objs)
            if self.logger:
                self.logger.log("ScorableRankBulkInserted", {"count": len(objs)})
            return [obj.id for obj in objs]
        return self._run(op)

    def get_by_query(self, query_text: str) -> List[ScorableRankORM]:
        """Get all ranks for a given query, ordered by score (descending)."""
        def op(s):
            return (
                s.query(ScorableRankORM)
                .filter_by(query_text=query_text)
                .order_by(ScorableRankORM.rank_score.desc())
                .all()
            )
        return self._run(op)

    def get_for_scorable(self, scorable_id: str, scorable_type: str) -> List[ScorableRankORM]:
        """Get all rank records for a given scorable (most recent first)."""
        def op(s):
            return (
                s.query(ScorableRankORM)
                .filter_by(scorable_id=scorable_id, scorable_type=scorable_type)
                .order_by(ScorableRankORM.created_at.desc())
                .all()
            )
        return self._run(op)
