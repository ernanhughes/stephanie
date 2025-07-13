# stephanie/memory/prompt_store.py
# stores/prompt_store.py
import json
import re
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import dialect
from sqlalchemy.orm import Session

from stephanie.models.goal import GoalORM
from stephanie.models.prompt import PromptORM


class PromptStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "prompt"

    def get_or_create_goal(
        self,
        goal_text: str,
        goal_type: str = None,
        focus_area: str = None,
        strategy: str = None,
        source: str = "user",
    ) -> GoalORM:
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
                    source=source,
                )
                self.session.add(goal)
                self.session.flush()  # Get ID before commit

                if self.logger:
                    self.logger.log(
                        "GoalCreated",
                        {
                            "goal_id": goal.id,
                            "goal_text": goal_text[:100],
                            "source": source,
                        },
                    )

            return goal

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("GoalGetOrCreateFailed", {"error": str(e)})
            raise

    def save(
        self,
        goal: dict,
        agent_name: str,
        prompt_key: str,
        prompt_text: str,
        response: Optional[str] = None,
        strategy: str = "default",
        pipeline_run_id: Optional[int] = None,
        extra_data: dict = None,
        version: int = 1,
    ):
        """
        Saves a prompt to the database and marks it as current for its key/agent.
        """
        try:
            goal_text = goal.get("goal_text", "")
            goal_type = goal.get("goal_type")
            # Get or create the associated goal
            goal_orm = self.get_or_create_goal(goal_text=goal_text, goal_type=goal_type)

            # Deactivate previous versions of this prompt key/agent combo
            self.session.query(PromptORM).filter_by(
                agent_name=agent_name, prompt_key=prompt_key
            ).update({"is_current": False})

            # Build ORM object
            db_prompt = PromptORM(
                goal_id=goal_orm.id,
                pipeline_run_id=pipeline_run_id,
                agent_name=agent_name,
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                response_text=response,
                strategy=strategy,
                version=version,
                extra_data=json.dumps(extra_data or {}),
            )

            self.session.add(db_prompt)
            self.session.flush()  # Get ID immediately

            if self.logger:
                self.logger.log(
                    "PromptStored",
                    {
                        "prompt_id": db_prompt.id,
                        "prompt_key": prompt_key,
                        "goal_id": goal_orm.id,
                        "agent": agent_name,
                        "strategy": strategy,
                        "length": len(prompt_text),
                        "timestamp": db_prompt.timestamp.isoformat(),
                    },
                )

            return db_prompt.id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log(
                    "PromptStoreFailed", {"error": str(e), "prompt_key": prompt_key}
                )
            raise

    def get_from_text(self, prompt_text: str) -> Optional[PromptORM]:
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

    def get_id_from_response(self, response_text: str) -> Optional[PromptORM]:
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
                    {"error": str(e)},
                )
            return None

    def find_matching(self, agent_name, prompt_text, strategy=None):
        query = self.session.query(PromptORM).filter_by(
            agent_name=agent_name, prompt_text=prompt_text
        )
        if strategy:
            query = query.filter_by(strategy=strategy)

        return [p.to_dict() for p in query.limit(10).all()]

    from difflib import SequenceMatcher

    def find_similar_prompt(
        self, agent_name, prompt_text, strategy=None, similarity_threshold=0.7
    ):
        def normalize(text):
            return re.sub(r"\s+", " ", text.strip().lower())

        text_a = normalize(prompt_text or "")

        query = self.session.query(PromptORM).filter(
            PromptORM.agent_name == agent_name,
            PromptORM.response_text.isnot(None),
            PromptORM.response_text != "",
        )

        if strategy:
            query = query.filter_by(strategy=strategy)

        candidates = query.limit(100).all()
        if not strategy:
            return [p.to_dict() for p in candidates]

        matches = []
        for p in candidates:
            text_b = normalize(p.prompt_text or "")
            if not text_a or not text_b:
                continue

            similarity = SequenceMatcher(None, text_a, text_b).ratio()
            if similarity >= similarity_threshold:
                matches.append((similarity, p))
            elif similarity >= 0.5:
                print(f"⚠️ Near miss ({similarity:.2f}): {text_b[:80]}")

        matches.sort(reverse=True, key=lambda x: x[0])
        return [p.to_dict() for similarity, p in matches]

    def get_prompt_training_set(self, goal: str, limit: int = 500) -> list[dict]:
        try:
            sql = text("""
                SELECT 
                    p.id,
                    g.goal_text AS goal,
                    p.prompt_text,
                    p.prompt_key,
                    p.timestamp,
                    h.text AS hypothesis_text,
                    h.elo_rating,
                    h.review
                FROM goals g
                JOIN prompts p ON p.goal_id = g.id
                JOIN hypotheses h ON h.prompt_id = p.id AND h.goal_id = g.id
                WHERE g.goal_text = :goal
                AND h.enabled = TRUE
                ORDER BY p.id, h.elo_rating DESC, h.updated_at DESC
                LIMIT :limit
            """)
            print("\n🔍 Final SQL Query:")
            print(
                sql.compile(
                    dialect=dialect(), compile_kwargs={"literal_binds": True}
                ).string
            )

            result = self.session.execute(sql, {"goal": goal, "limit": limit})

            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "goal": row[1],
                    "prompt_text": row[2],
                    "prompt_key": row[3],
                    "timestamp": row[4],
                    "hypothesis_text": row[5],
                    "elo_rating": row[6],
                    "review": row[7],
                }
                for row in rows
            ]

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log(
                    "GetLatestPromptsFailed", {"error": str(e), "goal": goal}
                )
            return []
