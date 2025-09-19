# stephanie/memory/method_plan_store.py
from __future__ import annotations

from typing import List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.method_plan import MethodPlanORM


class MethodPlanStore(BaseSQLAlchemyStore):
    orm_model = MethodPlanORM
    default_order_by = MethodPlanORM.id.desc()
    
    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "method_plans"

    def name(self) -> str:
        return self.name

    # -------------------
    # Insert
    # -------------------
    def add_method_plan(self, plan_data: dict) -> MethodPlanORM:
        """Insert a new method plan with validation on required fields."""
        def op(s):
            required_fields = ["idea_text"]
            missing = [f for f in required_fields if plan_data.get(f) is None]
            if missing:
                if self.logger:
                    self.logger.log(
                        "MissingRequiredFields",
                        {"missing_fields": missing, "raw_input": plan_data},
                    )
                raise ValueError(
                    f"Cannot save method plan. Missing required fields: {missing}"
                )

            plan = MethodPlanORM(**plan_data)
            s.add(plan)
            s.flush()
            return plan
        return self._run(op)

    # -------------------
    # Retrieval
    # -------------------
    def get_by_idea_text(self, idea_text: str) -> List[MethodPlanORM]:
        def op(s):
            return (
                s.query(MethodPlanORM)
                .filter(MethodPlanORM.idea_text.ilike(f"%{idea_text}%"))
                .all()
            )
        return self._run(op)

    def get_by_goal_id(self, goal_id: int) -> List[MethodPlanORM]:
        def op(s):
            return (
                s.query(MethodPlanORM)
                .filter(MethodPlanORM.goal_id == goal_id)
                .all()
            )
        return self._run(op)

    def get_top_scoring(self, limit: int = 5) -> List[MethodPlanORM]:
        def op(s):
            return (
                s.query(MethodPlanORM)
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
        return self._run(op)

    # -------------------
    # Update/Delete
    # -------------------
    def update_method_plan(self, plan_id: int, updates: dict) -> Optional[MethodPlanORM]:
        def op(s):
            plan = s.get(MethodPlanORM, plan_id)
            if not plan:
                raise ValueError(f"No method plan found with id {plan_id}")

            for key, value in updates.items():
                setattr(plan, key, value)

            s.flush()
            return plan
        return self._run(op)

    def delete_by_goal_id(self, goal_id: int) -> None:
        def op(s):
            s.query(MethodPlanORM).filter(
                MethodPlanORM.goal_id == goal_id
            ).delete()
        self._run(op)

    def clear_all(self) -> None:
        def op(s):
            s.query(MethodPlanORM).delete()
        self._run(op)
