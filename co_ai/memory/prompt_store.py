# stores/prompt_store.py
import json
from typing import Optional
from sqlalchemy.orm import Session
from co_ai.models.prompt import PromptORM
from co_ai.models.goal import GoalORM


class PromptStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "prompt"

    def get_or_create_goal(self, goal_text: str, goal_type: str = None,
                           focus_area: str = None, strategy: str = None,
                           source: str = "user") -> GoalORM:
        """
        Returns existing goal or creates a new one.
        """
        try:
            # Try to find by text
            goal = self.session.query(GoalORM).filter_by(goal_text=goal_text).first()
            if not goal:
                # Create new
                goal = GoalORM(
                    goal_text=goal_text,
                    goal_type=goal_type,
                    focus_area=focus_area,
                    strategy=strategy,
                    llm_suggested_strategy=None,
                    source=source
                )
                self.session.add(goal)
                self.session.flush()  # Get ID before commit

                if self.logger:
                    self.logger.log("GoalCreated", {
                        "goal_id": goal.id,
                        "goal_text": goal_text[:100],
                        "source": source
                    })

            return goal

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("GoalGetOrCreateFailed", {"error": str(e)})
            raise

    def save(self, goal: dict, agent_name: str, prompt_key: str, prompt_text: str,
             response: Optional[str] = None, strategy: str = "default",
             extra_data: dict = None, version: int = 1):
        """
        Saves a prompt to the database and marks it as current for its key/agent.
        """
        try:
            goal_text = goal.get("goal_text", "")
            goal_type=goal.get("goal_type")
            # Get or create the associated goal
            goal_orm = self.get_or_create_goal(goal_text=goal_text, goal_type=goal_type)

            # Deactivate previous versions of this prompt key/agent combo
            self.session.query(PromptORM).filter_by(
                agent_name=agent_name,
                prompt_key=prompt_key
            ).update({"is_current": False})

            # Build ORM object
            db_prompt = PromptORM(
                goal_id=goal_orm.id,
                agent_name=agent_name,
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                response_text=response,
                strategy=strategy,
                version=version,
                extra_data=json.dumps(extra_data or {})
            )

            self.session.add(db_prompt)
            self.session.flush()  # Get ID immediately

            if self.logger:
                self.logger.log("PromptStored", {
                    "prompt_id": db_prompt.id,
                    "prompt_key": prompt_key,
                    "goal_id": goal_orm.id,
                    "agent": agent_name,
                    "strategy": strategy,
                    "length": len(prompt_text),
                    "timestamp": db_prompt.timestamp.isoformat()
                })

            return db_prompt.id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log(
                    "PromptStoreFailed", {"error": str(e), "prompt_key": prompt_key}
                )
            raise

    def get_from_text(
        self,
        prompt_text: str
    ) -> Optional[PromptORM]:
        """
        Retrieve a prompt from the DB based on its exact prompt_text.
        Optionally filter by agent_name and/or strategy.
        """
        try:
            query = self.session.query(PromptORM).filter(
                PromptORM.prompt_text == prompt_text
            )

            prompt = query.order_by(PromptORM.timestamp.desc()).first()

            if self.logger:
                self.logger.log(
                    "PromptLookup",
                    {
                        "matched": bool(prompt),
                        "text_snippet": prompt_text[:100],
                    },
                )

            return prompt

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log(
                    "PromptLookupFailed",
                    {"error": str(e), "text_snippet": prompt_text[:100]},
                )
            return None

    def get_id_from_response(
        self,
        response_text: str
    ) -> Optional[PromptORM]:
        """
        Retrieve a prompt from the DB based on its exact prompt_text.
        Optionally filter by agent_name and/or strategy.
        """
        try:
            query = self.session.query(PromptORM).filter(
                PromptORM.response_text == response_text
            )

            prompt = query.order_by(PromptORM.timestamp.desc()).first()

            if self.logger:
                self.logger.log(
                    "PromptLookup",
                    {
                        "matched": bool(prompt),
                        "text_snippet": response_text[:100],
                    },
                )

            return prompt.id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log(
                    "PromptLookupFailed",
                    {"error": str(e), "text_snippet": prompt_text[:100]},
                )
            return None

    def find_matching(self, agent_name, prompt_text, strategy=None):
        query = self.session.query(PromptORM).filter_by(
            agent_name=agent_name,
            prompt_text=prompt_text
        )
        if strategy:
            query = query.filter_by(strategy=strategy)

        return [p.to_dict() for p in query.limit(10).all()]
