# stephanie/memory/goal_dimensions_store.py
from __future__ import annotations

from typing import List

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.goal_dimension import GoalDimensionORM


class GoalDimensionsStore(BaseSQLAlchemyStore):
    orm_model = GoalDimensionORM
    default_order_by = GoalDimensionORM.rank.asc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "goal_dimensions"

    def name(self) -> str:
        return self.name

    # -------------------
    # Insert
    # -------------------
    def insert(self, row: dict) -> GoalDimensionORM:
        """Insert a single GoalDimension row."""
        def op():
            obj = GoalDimensionORM(**row)
            s = self._scope()
            s.add(obj)
            s.flush()
            return obj
        return self._run(op)

    # -------------------
    # Retrieval
    # -------------------
    def find_by_goal_id(self, goal_id: int) -> List[dict]:
        """Return dimensions for a single goal, ordered by rank."""
        def op():
            results = (
                self._scope()
                .query(GoalDimensionORM)
                .filter_by(goal_id=goal_id)
                .order_by(GoalDimensionORM.rank.asc())
                .all()
            )
            return [r.to_dict() for r in results]
        return self._run(op)

    def find_by_goal_ids(self, goal_ids: List[int]) -> List[dict]:
        """Return dimensions for multiple goals, grouped/ordered by goal_id then rank."""
        def op():
            results = (
                self._scope()
                .query(GoalDimensionORM)
                .filter(GoalDimensionORM.goal_id.in_(goal_ids))
                .order_by(GoalDimensionORM.goal_id.asc(), GoalDimensionORM.rank.asc())
                .all()
            )
            return [r.to_dict() for r in results]
        return self._run(op)
