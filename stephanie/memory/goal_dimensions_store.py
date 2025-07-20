from stephanie.models.goal_dimension import GoalDimensionORM


class GoalDimensionsStore:
    def __init__(self, session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "goal_dimensions"

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
