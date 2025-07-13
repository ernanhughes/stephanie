# stephanie/memory/reflection_delta_store.py
from typing import Optional

from stephanie.memory.base import BaseStore
from stephanie.models.reflection_delta import ReflectionDeltaORM


class ReflectionDeltaStore(BaseStore):
    def __init__(self, db, logger=None):
        super().__init__(db, logger)
        self.name = "reflection_deltas"

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "reflection_deltas"

    def insert(self, delta: ReflectionDeltaORM) -> int:
        try:
            self.db.add(delta)
            self.db.flush()
            self.db.commit()

            if self.logger:
                self.logger.log(
                    "ReflectionDeltaInserted",
                    {
                        "delta_id": delta.id,
                        "goal_id": delta.goal_id,
                        "run_id_a": delta.run_id_a,
                        "run_id_b": delta.run_id_b,
                        "score_a": delta.score_a,
                        "score_b": delta.score_b,
                        "score_delta": delta.score_delta,
                        "strategy_diff": delta.strategy_diff,
                        "model_diff": delta.model_diff,
                    },
                )

            return delta.id

        except Exception as e:
            self.db.rollback()
            if self.logger:
                self.logger.log("ReflectionDeltaInsertFailed", {"error": str(e)})
            raise

    def get_by_goal_id(self, goal_id: int) -> list[ReflectionDeltaORM]:
        try:
            return (
                self.db.query(ReflectionDeltaORM)
                .filter_by(goal_id=goal_id)
                .order_by(ReflectionDeltaORM.created_at.desc())
                .all()
            )
        except Exception as e:
            if self.logger:
                self.logger.log("ReflectionDeltasFetchFailed", {"error": str(e)})
            return []

    def get_by_run_ids(
        self, run_id_a: str, run_id_b: str
    ) -> Optional[ReflectionDeltaORM]:
        try:
            return (
                self.db.query(ReflectionDeltaORM)
                .filter_by(run_id_a=run_id_a, run_id_b=run_id_b)
                .first()
            )
        except Exception as e:
            if self.logger:
                self.logger.log("ReflectionDeltaFetchFailed", {"error": str(e)})
            return None

    def get_all(self, limit: int = 100) -> list[ReflectionDeltaORM]:
        try:
            return (
                self.db.query(ReflectionDeltaORM)
                .order_by(ReflectionDeltaORM.created_at.desc())
                .limit(limit)
                .all()
            )
        except Exception as e:
            if self.logger:
                self.logger.log("ReflectionDeltasFetchFailed", {"error": str(e)})
            return []

    def find(self, filters: dict) -> list[ReflectionDeltaORM]:
        try:
            query = self.db.query(ReflectionDeltaORM)

            if "goal_id" in filters:
                query = query.filter(ReflectionDeltaORM.goal_id == filters["goal_id"])

            if "run_id_a" in filters and "run_id_b" in filters:
                query = query.filter(
                    ReflectionDeltaORM.run_id_a == filters["run_id_a"],
                    ReflectionDeltaORM.run_id_b == filters["run_id_b"],
                )

            if "score_delta_gt" in filters:
                query = query.filter(
                    ReflectionDeltaORM.score_delta > filters["score_delta_gt"]
                )

            if "strategy_diff" in filters:
                query = query.filter(
                    ReflectionDeltaORM.strategy_diff == filters["strategy_diff"]
                )

            return query.order_by(ReflectionDeltaORM.created_at.desc()).all()

        except Exception as e:
            if self.logger:
                self.logger.log("ReflectionDeltasFetchFailed", {"error": str(e)})
            return []
