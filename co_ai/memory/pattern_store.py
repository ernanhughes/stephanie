# stores/pattern_stat_store.py
from co_ai.models.pattern_stat import PatternStatORM
from datetime import datetime


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
                self.logger.log("PatternStatsStored", {
                    "goal_id": stats[0].goal_id,
                    "hypothesis_id": stats[0].hypothesis_id,
                    "agent": stats[0].agent_name,
                    "model": stats[0].model_name,
                    "count": len(stats),
                    "timestamp": datetime.utcnow().isoformat()
                })

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("PatternStatsInsertFailed", {"error": str(e)})
            raise