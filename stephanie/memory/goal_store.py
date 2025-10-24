# stephanie/memory/goal_store.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.exc import IntegrityError

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.goal import GoalORM


class GoalStore(BaseSQLAlchemyStore):
    orm_model = GoalORM
    default_order_by = GoalORM.created_at.desc()
    
    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "goals"

    # -------------------
    # Retrieval
    # -------------------
    def get_from_text(self, goal_text: str) -> Optional[GoalORM]:
        """Return the first goal matching the exact text."""
        def op(s):
            return s.query(GoalORM).filter(GoalORM.goal_text == goal_text).first()
        return self._run(op)

    def get_all(self) -> List[GoalORM]:
        """Return all goals."""
        def op(s):
            return s.query(GoalORM).all()
        return self._run(op)

    def get_by_id(self, goal_id: int) -> Optional[GoalORM]:
        """Return a single goal by ID."""
        def op(s):
            return s.query(GoalORM).filter(GoalORM.id == goal_id).first()
        return self._run(op)

    # -------------------
    # Insert / Upsert
    # -------------------
    def create(self, goal_dict: dict) -> GoalORM:
        """
        Create a new goal. Rolls back and re-queries if duplicate.
        """
        def op(s):
            try:
                new_goal = GoalORM(
                    goal_text=goal_dict["goal_text"],
                    goal_type=goal_dict.get("goal_type"),
                    focus_area=goal_dict.get("focus_area"),
                    strategy=goal_dict.get("strategy"),
                    llm_suggested_strategy=goal_dict.get("llm_suggested_strategy"),
                    source=goal_dict.get("source", "user"),
                    created_at=goal_dict.get("created_at") or datetime.now(timezone.utc),
                )
                s.add(new_goal)
                s.flush()
                if self.logger:
                    self.logger.log(
                        "GoalCreated",
                        {
                            "goal_id": new_goal.id,
                            "goal_text": new_goal.goal_text[:100],
                            "source": new_goal.source,
                        },
                    )
                return new_goal
            except IntegrityError:
                s.rollback()
                return self.get_from_text(goal_dict["goal_text"])
        return self._run(op)

    def get_or_create(self, goal_dict: dict) -> GoalORM:
        """
        Return existing goal or create a new one.
        """
        goal_text = goal_dict.get("goal_text")
        if not goal_text:
            raise ValueError("Missing 'goal_text' in input")

        existing = self.get_from_text(goal_text)
        if existing:
            return existing
        return self.create(goal_dict)
