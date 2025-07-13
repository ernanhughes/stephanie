# stephanie/memory/method_plan_store.py
# stores/method_plan_store.py
from sqlalchemy.orm import Session

from stephanie.memory import BaseStore
from stephanie.models.method_plan import MethodPlanORM


class MethodPlanStore(BaseStore):
    def __init__(self, session, logger):
        super().__init__(session, logger)
        self.name = "method_plans"

    def name(self) -> str:
        return "method_plans"

    def add_method_plan(self, plan_data: dict) -> MethodPlanORM:
        """
        Adds a new method plan to the database.

        Args:
            plan_data (dict): Must include all fields from MethodPlanORM

        Returns:
            MethodPlanORM: The saved ORM object
        """
        required_fields = ["idea_text"]
        missing = [f for f in required_fields if plan_data.get(f) is None]

        if missing:
            self.logger.log(
                "MissingRequiredFields",
                {"missing_fields": missing, "raw_input": plan_data},
            )
            raise ValueError(
                f"Cannot save method plan. Missing required fields: {missing}"
            )

        plan = MethodPlanORM(**plan_data)
        self.db.add(plan)
        self.db.commit()
        self.db.refresh(plan)
        return plan

    def get_by_idea_text(self, idea_text: str) -> list[MethodPlanORM]:
        """
        Retrieves all method plans generated from a specific idea

        Args:
            idea_text (str): Text of the idea used to generate this method

        Returns:
            List of MethodPlanORM objects
        """
        return (
            self.db.query(MethodPlanORM)
            .filter(MethodPlanORM.idea_text.ilike(f"%{idea_text}%"))
            .all()
        )

    def get_by_goal_id(self, goal_id: int) -> list[MethodPlanORM]:
        """
        Retrieves all method plans linked to a specific research goal

        Args:
            goal_id (int): GoalORM.id

        Returns:
            List of MethodPlanORM objects
        """
        return (
            self.db.query(MethodPlanORM).filter(MethodPlanORM.goal_id == goal_id).all()
        )

    def get_top_scoring(self, limit: int = 5) -> list[MethodPlanORM]:
        """
        Get top N method plans ranked by composite score

        Returns:
            List of MethodPlanORM objects
        """
        return (
            self.db.query(MethodPlanORM)
            .order_by(
                (
                    MethodPlanORM.score_novelty * 0.3
                    + MethodPlanORM.score_feasibility * 0.2
                    + MethodPlanORM.score_impact * 0.3
                    + MethodPlanORM.score_alignment * 0.2
                ).desc()
            )
            .limit(limit)
            .all()
        )

    def update_method_plan(self, plan_id: int, updates: dict) -> MethodPlanORM:
        """
        Updates an existing method plan with new values

        Args:
            plan_id (int): ID of the plan to update
            updates (dict): Fields to update (score_novelty, code_plan, etc.)

        Returns:
            Updated MethodPlanORM object
        """
        plan = self.db.query(MethodPlanORM).get(plan_id)
        if not plan:
            raise ValueError(f"No method plan found with id {plan_id}")

        for key, value in updates.items():
            setattr(plan, key, value)

        self.db.commit()
        self.db.refresh(plan)
        return plan

    def delete_by_goal_id(self, goal_id: int) -> None:
        """
        Delete all method plans associated with a given goal

        Args:
            goal_id (int): GoalORM.id
        """
        self.db.query(MethodPlanORM).filter(MethodPlanORM.goal_id == goal_id).delete()
        self.db.commit()

    def clear_all(self) -> None:
        """
        Clear all method plans — useful for testing
        """
        self.db.query(MethodPlanORM).delete()
        self.db.commit()
