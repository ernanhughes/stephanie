import json

from co_ai.memory import BaseStore
from co_ai.models.sharpening_result import SharpeningResult


class MRQStore(BaseStore):
    def __init__(self, db, cfg, embeddings, logger=None):
        super().__init__(db, logger)
        self.cfg = cfg
        self.embeddings = embeddings
        self.name = "mrq"

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "mrq"

    def log_evaluations(self):
        return self.cfg.get("log_evaluations", False)

    def add(
        self,
        goal: str,
        strategy: str,
        prompt: str,
        response: str,
        reward: float,
        metadata: dict = None,
    ):
        """Add a new (prompt, response, reward) to MRQ memory"""
        emb_prompt = self.embeddings.get_or_create(prompt)

        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
            INSERT INTO mrq_memory (goal, strategy, prompt, response, reward, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
                    (
                        goal,
                        strategy,
                        prompt,
                        response,
                        reward,
                        emb_prompt,
                        json.dumps(metadata or {}),
                    ),
                )
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log(
                    "PromptLookupFailed",
                    {"error": str(e), "prompt_snippet": prompt[:100]},
                )

    def get_similar_prompt(self, prompt: str, top_k=5):
        """Retrieve most similar prompt-response pairs"""
        emb_query = self.embeddings.get_or_create(prompt)

        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
            SELECT strategy, prompt, response, reward
            FROM mrq_memory
            ORDER BY embedding <-> %s
            LIMIT %s
            """,
                    (str(emb_query), top_k),
                )

            return [
                {"strategy": r[0], "prompt": r[1], "response": r[2], "reward": r[3]}
                for r in cur.fetchall()
            ]
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log(
                    "PromptLookupFailed",
                    {"error": str(e), "prompt_snippet": prompt[:100]},
                )
            return {}

    def get_similar_prompts(self, prompt: str, top_k=5):
        emb = self.embeddings.get_or_create(prompt)

        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
            SELECT prompt, response, reward, strategy
            FROM mrq_memory
            ORDER BY prompt_embedding <-> %s
            LIMIT %s
            """,
                    (str(emb), top_k),
                )

            return cur.fetchall()
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log(
                    "PromptLookupFailed",
                    {"error": str(e), "prompt_snippet": prompt[:100]},
                )
            return []

    def insert_sharpening_prediction(self, prediction: dict, goal_text: str=None):
        """
        Insert a sharpening prediction into the database.
        Resolves the goal_id internally based on goal_text.

        :param prediction: dict containing prompt, output_a, output_b, preferred, predicted, scores
        :param goal_text: str - raw goal string to look up goal_id
        """
        try:
            with self.db.cursor() as cur:
                goal_id = None
                if goal_text:
                    # Resolve goal_id
                    cur.execute(
                        "SELECT id FROM goals WHERE goal_text = %s LIMIT 1", (goal_text,)
                    )
                    result = cur.fetchone()
                    if not result:
                        raise ValueError(f"Goal not found: {goal_text}")
                    goal_id = result[0]

                # Insert prediction
                cur.execute(
                    """
                    INSERT INTO sharpening_predictions (
                        goal_id, prompt_text, output_a, output_b,
                        preferred, predicted, value_a, value_b
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """,
                    (
                        goal_id,
                        prediction["prompt_text"],
                        prediction["output_a"],
                        prediction["output_b"],
                        prediction["preferred"],
                        prediction["predicted"],
                        prediction["value_a"],
                        prediction["value_b"],
                    ),
                )
                self.db.commit()

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log(
                    "InsertSharpeningPredictionFailed",
                    {"error": str(e), "goal_text": goal_text, "prediction": prediction},
                )

    def get_training_pairs(
        self, goal: str, limit: int = 100, agent_name="generation"
    ) -> list[dict]:
        try:
            with self.db.cursor() as cur:
                query = """
                    WITH top_h AS (
                        SELECT DISTINCT ON (p.id)
                            p.id AS prompt_id,
                            g.goal_text AS goal,
                            p.prompt_text,
                            h.text AS output_a,
                            h.elo_rating AS rating_a
                        FROM prompts p
                        JOIN goals g ON p.goal_id = g.id
                        JOIN hypotheses h ON h.prompt_id = p.id
                        WHERE h.enabled = TRUE
                        AND h.goal_id = g.id
                        AND p.agent_name = %s
                        AND g.goal_text = %s
                        ORDER BY p.id, h.elo_rating DESC
                    ),
                    bottom_h AS (
                        SELECT DISTINCT ON (p.id)
                            p.id AS prompt_id,
                            h.text AS output_b,
                            h.elo_rating AS rating_b
                        FROM prompts p
                        JOIN hypotheses h ON h.prompt_id = p.id
                        JOIN goals g ON p.goal_id = g.id
                        WHERE h.enabled = TRUE
                        AND h.goal_id = g.id
                        AND p.agent_name = %s
                        AND g.goal_text = %s
                        ORDER BY p.id, h.elo_rating ASC
                    )
                    SELECT 
                        top_h.prompt_id,
                        top_h.goal,
                        top_h.prompt_text,
                        top_h.output_a,
                        top_h.rating_a,
                        bottom_h.output_b,
                        bottom_h.rating_b
                    FROM top_h
                    JOIN bottom_h ON top_h.prompt_id = bottom_h.prompt_id
                    WHERE top_h.rating_a != bottom_h.rating_b
                    LIMIT %s;
                """
                params = (agent_name, goal, agent_name, goal, limit)  # 🛠️ FIXED

                if self.logger:
                    full_sql = cur.mogrify(query, params).decode("utf-8")
                    self.logger.log("SQLQuery", {"query": full_sql})

                cur.execute(query, params)
                rows = cur.fetchall()

                return [
                    {
                        "prompt": row[2],
                        "output_a": row[3],
                        "output_b": row[5],
                        "preferred": "a" if row[4] > row[6] else "b",
                        "rating_a": row[4],
                        "rating_b": row[6],
                    }
                    for row in rows
                ]

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log(
                    "GetMRQTrainingPairsFailed", {"error": str(e), "goal": goal}
                )
            return []

    def insert_sharpening_result(self, result: SharpeningResult):
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sharpening_results (
                    goal, prompt, template, original_output, sharpened_output,
                    preferred_output, winner, improved, comparison,
                    score_a, score_b, score_diff, best_score,
                    prompt_template
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    result.goal,
                    result.prompt,
                    result.template,
                    result.original_output,
                    result.sharpened_output,
                    result.preferred_output,
                    result.winner,
                    result.improved,
                    result.comparison,
                    result.score_a,
                    result.score_b,
                    result.score_diff,
                    result.best_score,
                    result.prompt_template,
                ),
            )
            self.db.commit()
