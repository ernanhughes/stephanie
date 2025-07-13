# stephanie/memory/pattern_store.py
# stores/pattern_stat_store.py
from datetime import datetime

from stephanie.models.pattern_stat import PatternStatORM


class PatternStatStore:
    def __init__(self, session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "pattern_stats"

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
