import json

from co_ai.memory import BaseStore


class PromptStore(BaseStore):
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger
        self.name = "prompt"

    def name(self) -> str:
        return "prompt"

    def save(
        self,
        agent_name,
        prompt_key,
        prompt_text,
        response=None,
        strategy="default",
        version=1,
    ):
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prompts (
                        agent_name, prompt_key, prompt_text, response_text,
                        source, version, is_current, strategy, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, TRUE, %s, %s)
                    """,
                    (
                        agent_name,
                        prompt_key,
                        prompt_text,
                        response,
                        "manual",
                        version,
                        strategy,
                        json.dumps({}),
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
            if self.logger:
                self.logger.log("PromptFetchFailed", {"error": str(e)})
            return []
