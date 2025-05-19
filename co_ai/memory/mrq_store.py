from co_ai.memory import BaseStore
import json


class MRQStore(BaseStore):
    def __init__(self, db, embeddings, logger=None):
        self.db = db
        self.embeddings = embeddings
        self.logger = logger
        self.name = "mrq"

    def name(self) -> str:
        return "mrq"

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
            if self.logger:
                self.logger.log(
                    "PromptLookupFailed",
                    {"error": str(e), "prompt_snippet": prompt[:100]},
                )
            return []

    def insert_sharpening_prediction(self, prediction: dict, goal_text: str):
        """
        Insert a sharpening prediction into the database.
        Resolves the goal_id internally based on goal_text.

        :param prediction: dict containing prompt, output_a, output_b, preferred, predicted, scores
        :param goal_text: str - raw goal string to look up goal_id
        """
        try:
            with self.db.cursor() as cur:
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
                        prediction["prompt"],
                        prediction["output_a"],
                        prediction["output_b"],
                        prediction["preferred"],
                        prediction["predicted"],
                        prediction["scores"]["value_a"],
                        prediction["scores"]["value_b"],
                    ),
                )

                self.db.commit()

        except Exception as e:
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
                    SELECT 
                        p.id, 
                        g.goal_text AS goal, 
                        p.prompt_text, 
                        h1.text AS output_a, 
                        h1.elo_rating AS rating_a,
                        h2.text AS output_b, 
                        h2.elo_rating AS rating_b
                    FROM prompts p
                    JOIN goals g ON p.goal_id = g.id
                    JOIN hypotheses h1 ON h1.prompt_id = p.id
                    JOIN hypotheses h2 ON h2.prompt_id = p.id AND h1.id != h2.id
                    WHERE g.goal_text = %s
                    AND p.agent_name = %s
                    AND h1.enabled = TRUE AND h2.enabled = TRUE
                    AND h1.goal_id = g.id AND h2.goal_id = g.id
                    AND h1.elo_rating != h2.elo_rating
                    ORDER BY p.id, GREATEST(h1.elo_rating, h2.elo_rating) DESC
                    LIMIT %s;
                """
                params = (goal, agent_name, limit)

                # DEBUG: Print SQL query with full values
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
            if self.logger:
                self.logger.log(
                    "GetMRQTrainingPairsFailed", {"error": str(e), "goal": goal}
                )
            return []
