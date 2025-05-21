import json

from co_ai.memory import BaseStore


class PromptStore(BaseStore):
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger
        self.name = "prompt"

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "prompt"

    def get_or_create_goal(self, goal_text, goal_type=None, focus_area=None,
                           strategy=None, source="user"):
        """
        Looks up or inserts a goal in the `goals` table and returns the goal_id.
        Assumes `conn` is a psycopg2 or asyncpg connection or cursor.
        """
        try:
            with self.db.cursor() as cur:
                # Try to find existing goal
                cur.execute(
                    """
                    SELECT id FROM goals
                    WHERE goal_text = %s
                    LIMIT 1
                """,
                    (goal_text,),
                )
                result = cur.fetchone()
                if result:
                    return result[0]

                # Insert new goal
                cur.execute(
                    """
                    INSERT INTO goals (goal_text, goal_type, focus_area, strategy, source)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (goal_text, goal_type, focus_area, strategy, source),
                )
                new_id = cur.fetchone()[0]
                return new_id
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("GoalGetFailed", {"error": str(e)})
            return None


    def save(
        self,
        goal: str,
        agent_name,
        prompt_key,
        prompt_text,
        response=None,
        strategy="default",
        version=1,
        meta_data=None
    ):
        try:
            goal_id =  self.get_or_create_goal(goal)
            if meta_data is None:
                meta_data = {}
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prompts (
                        goal_id, agent_name, prompt_key, prompt_text, response_text,
                        source, version, is_current, strategy, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, %s, %s)
                    """,
                    (
                        goal_id,
                        agent_name,
                        prompt_key,
                        prompt_text,
                        response,
                        "manual",
                        version,
                        strategy,
                        json.dumps(meta_data),
                    ),
                )
                cur.execute(
                    """
                    UPDATE prompts SET is_current = FALSE
                    WHERE agent_name = %s AND prompt_key = %s AND is_current IS TRUE
                    """,
                    (agent_name, prompt_key),
                )
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("PromptLogFailed", {"error": str(e)})

    def store_evaluation(
        self,
        prompt_id: int,
        benchmark_name: str,
        score: float,
        metrics: dict,
        evaluator: str = "auto",
        notes: str | None = None,
        dataset_hash: list | None = None,
    ):
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                        INSERT INTO prompt_evaluations
                        (prompt_id, benchmark_name, score, metrics, evaluator, notes, dataset_hash)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                    (
                        prompt_id,
                        benchmark_name,
                        score,
                        json.dumps(metrics),
                        evaluator,
                        notes,
                        dataset_hash,
                    ),
                )
            if self.logger:
                self.logger.log(
                    "PromptEvaluationStored",
                    {
                        "prompt_id": prompt_id,
                        "benchmark_name": benchmark_name,
                        "score": score,
                    },
                )
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log(
                    "PromptEvaluationStoreFailed",
                    {"prompt_id": prompt_id, "error": str(e)},
                )

    def get_latest_prompts(self, limit: int = 10) -> list[dict]:
        """
        Return the latest prompts added to the database, ordered by most recent.

        Args:
            limit (int): Number of prompts to return.

        Returns:
            List of prompt records as dictionaries.
        """
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, agent_name, prompt_key, prompt_text, response_text,
                           source, version, is_current, strategy, metadata, timestamp
                    FROM prompts
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("PromptFetchFailed", {"error": str(e)})
            return []

    def get_prompt_training_set(self, goal: str, limit: int = 5, agent_name='generation') -> list[dict]:
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (p.id)
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
                    WHERE g.goal_text = %s
                    AND p.agent_name = %s
                    AND h.enabled = TRUE
                    ORDER BY p.id, h.elo_rating DESC, h.updated_at DESC
                    LIMIT %s
                    """,
                    (goal, agent_name, limit),
                )
                rows = cur.fetchall()
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
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("GetLatestPromptsFailed", {
                    "error": str(e),
                    "goal": goal
                })
            return []

    def get_eliciting_prompts(self, goal: str, limit: int = 20, agent_name='generation') -> list[dict]:
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    SELECT p.id, p.prompt_text, p.prompt_key, p.timestamp, e.score
                    FROM prompts p
                    LEFT JOIN LATERAL (
                        SELECT score
                        FROM prompt_evaluations e
                        WHERE e.prompt_id = p.id
                        ORDER BY e.timestamp DESC
                        LIMIT 1
                    ) e ON true
                    WHERE p.agent_name = %s
                    AND p.prompt_text IS NOT NULL
                    AND p.goal = %s
                    ORDER BY e.score DESC NULLS LAST, p.timestamp DESC
                    LIMIT %s
                    """,
                    (agent_name, goal, limit),
                )
                rows = cur.fetchall()
                return [
                    {
                        "id": row[0],
                        "prompt_text": row[1],
                        "prompt_key": row[2],
                        "timestamp": row[3],
                        "score": row[4],
                    }
                    for row in rows
                ]
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("GetElicitingPromptsFailed", {"error": str(e), "goal": goal})
            return []
