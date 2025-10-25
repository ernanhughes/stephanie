# stephanie/memory/reflection_delta_store.py
from __future__ import annotations

from typing import Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.reflection_delta import ReflectionDeltaORM


class ReflectionDeltaStore(BaseSQLAlchemyStore):
    orm_model = ReflectionDeltaORM
    default_order_by = ReflectionDeltaORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "reflection_deltas"

    def insert(self, delta: ReflectionDeltaORM) -> int:
        def op(s):
            
            s.add(delta)
            s.flush()
            if self.logger:
                self.logger.log("ReflectionDeltaInserted", {
                    "delta_id": delta.id,
                    "goal_id": delta.goal_id,
                    "run_id_a": delta.run_id_a,
                    "run_id_b": delta.run_id_b,
                    "score_a": delta.score_a,
                    "score_b": delta.score_b,
                    "score_delta": delta.score_delta,
                    "strategy_diff": delta.strategy_diff,
                    "model_diff": delta.model_diff,
                })
            return delta.id
        return self._run(op)

    def get_by_goal_id(self, goal_id: int) -> list[ReflectionDeltaORM]:
        def op(s):
            
            return (
                s.query(ReflectionDeltaORM)
                .filter_by(goal_id=goal_id)
                .order_by(ReflectionDeltaORM.created_at.desc())
                .all()
            )
        return self._run(op)

    def get_by_run_ids(self, run_id_a: str, run_id_b: str) -> Optional[ReflectionDeltaORM]:
        def op(s):
            
            return (
                s.query(ReflectionDeltaORM)
                .filter_by(run_id_a=run_id_a, run_id_b=run_id_b)
                .first()
            )
        return self._run(op)

    def get_all(self, limit: int = 100) -> list[ReflectionDeltaORM]:
        def op(s):
            
            return (
                s.query(ReflectionDeltaORM)
                .order_by(ReflectionDeltaORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def find(self, filters: dict) -> list[ReflectionDeltaORM]:
        def op(s):
            
            query = s.query(ReflectionDeltaORM)

            if "goal_id" in filters:
                query = query.filter(ReflectionDeltaORM.goal_id == filters["goal_id"])
            if "run_id_a" in filters and "run_id_b" in filters:
                query = query.filter(
                    ReflectionDeltaORM.run_id_a == filters["run_id_a"],
                    ReflectionDeltaORM.run_id_b == filters["run_id_b"],
                )
            if "score_delta_gt" in filters:
                query = query.filter(ReflectionDeltaORM.score_delta > filters["score_delta_gt"])
            if "strategy_diff" in filters:
                query = query.filter(ReflectionDeltaORM.strategy_diff == filters["strategy_diff"])

            return query.order_by(ReflectionDeltaORM.created_at.desc()).all()
        return self._run(op)
