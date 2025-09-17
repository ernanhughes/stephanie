# stephanie/memory/goal_dimensions_store.py
from __future__ import annotations

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.goal_dimension import GoalDimensionORM


class GoalDimensionsStore(BaseSQLAlchemyStore):
    orm_model = GoalDimensionORM
    default_order_by = GoalDimensionORM.rank.asc()

    def __init__(self, session, logger=None):
        super().__init__(session, logger)
        self.name = "goal_dimensions"

    def name(self) -> str:
        return self.name

    def insert(self, row: dict) -> GoalDimensionORM:
        obj = GoalDimensionORM(**row)
        self.session.add(obj)
        self.session.commit()
        return obj

    def find_by_goal_id(self, goal_id: int) -> list[dict]:
        results = (
            self.session.query(GoalDimensionORM)
            .filter_by(goal_id=goal_id)
            .order_by(GoalDimensionORM.rank.asc())
            .all()
        )
        return [r.to_dict() for r in results]

    def find_by_goal_ids(self, goal_ids: list[int]) -> list[dict]:
        results = (
            self.session.query(GoalDimensionORM)
            .filter(GoalDimensionORM.goal_id.in_(goal_ids))
            .order_by(GoalDimensionORM.goal_id.asc(), GoalDimensionORM.rank.asc())
            .all()
        )
        return [r.to_dict() for r in results]
