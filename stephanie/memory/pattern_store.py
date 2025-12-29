# stephanie/memory/pattern_store.py
from __future__ import annotations

from typing import List

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.pattern_stat import PatternStatORM


class PatternStatStore(BaseSQLAlchemyStore):
    orm_model = PatternStatORM
    default_order_by = PatternStatORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "pattern_stats"

    def insert(self, stats: List[PatternStatORM]) -> None:
        """Insert multiple pattern stats in a single transaction"""
        def op(s):
            s.bulk_save_objects(stats)
            if self.logger:
                self.logger.log(
                    "PatternStatsStored",
                    {"count": len(stats), "ids": [getattr(x, "id", None) for x in stats]},
                )
        return self._run(op)
