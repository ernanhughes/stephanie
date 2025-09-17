# stephanie/memory/pattern_store.py
from __future__ import annotations

from datetime import datetime

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.pattern_stat import PatternStatORM


class PatternStatStore(BaseSQLAlchemyStore):
    orm_model = PatternStatORM
    default_order_by = PatternStatORM.created_at.desc()

    def __init__(self, session, logger=None):
        super().__init__(session, logger)
        self.name = "pattern_stats"

    def name(self) -> str:
        return self.name

    def insert(self, stats: list[PatternStatORM]):
        """Insert multiple pattern stats at once"""
        try:
            self.session.bulk_save_objects(stats)
            self.session.commit()

            if self.logger:
                self.logger.log("PatternStatsStored", {"stats": stats})

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("PatternStatsInsertFailed", {"error": str(e)})
            raise
