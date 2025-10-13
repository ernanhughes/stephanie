# stephanie/memory/reasoning_sample_store.py
from __future__ import annotations
from typing import List
from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.reasoning_sample import ReasoningSampleORM

class ReasoningSampleStore(BaseSQLAlchemyStore):
    """
    Read-only store for reasoning_samples_view.
    Used by data loaders (TinyRecursion, SICQL, etc.)
    to fetch structured reasoning examples.
    """
    orm_model = ReasoningSampleORM
    default_order_by = ReasoningSampleORM.created_at.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "reasoning_samples"

    def get_all(self, limit: int = 1000) -> List[ReasoningSampleORM]:
        def op(s):
            return (
                s.query(ReasoningSampleORM)
                .order_by(ReasoningSampleORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)


    def get_by_target_type(self, target_type: str, limit: int = 100) -> List[ReasoningSampleORM]:
        def op(s):
            return (
                s.query(ReasoningSampleORM)
                .filter(ReasoningSampleORM.scorable_type == target_type)
                .order_by(ReasoningSampleORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def get_by_goal(self, goal_text: str, limit: int = 50) -> List[ReasoningSampleORM]:
        def op(s):
            return (
                s.query(ReasoningSampleORM)
                .filter(ReasoningSampleORM.goal_text.ilike(f"%{goal_text}%"))
                .limit(limit)
                .all()
            )
        return self._run(op)
