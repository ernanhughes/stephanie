# stephanie/memory/goal_store.py

from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stephanie.models.goal import GoalORM


class GoalStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "goals"

    def name(self) -> str:
        return "goals"

    def get_from_text(self, goal_text: str):
        return (
            self.session.query(GoalORM).filter(GoalORM.goal_text == goal_text).first()
        )

    def create(self, goal_dict: dict):
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
            self.session.add(new_goal)
            self.session.commit()
            self.session.refresh(new_goal)

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
            self.session.rollback()
            return self.get_by_text(goal_dict["goal_text"])

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("GoalCreateFailed", {"error": str(e)})
            raise

    def get_or_create(self, goal_dict: dict):
        """
        Returns existing goal or creates a new one.
        """
        goal_text = goal_dict.get("goal_text")
        if not goal_text:
            raise ValueError("Missing 'goal_text' in input")

        existing = self.get_from_text(goal_text)
        if existing:
            return existing

        return self.create(goal_dict)
