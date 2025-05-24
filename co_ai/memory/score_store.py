import json
from typing import Optional, List, Dict
from datetime import datetime
from co_ai.models.score import Score
from co_ai.memory.base_store import BaseStore


class ScoreStore(BaseStore):
    def __init__(self, db, logger=None):
        super().__init__(db, logger)
        self.name = "scores"
        self.table_name = "scores"

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "scores"

    def insert(self, score: Score):
        """
        Inserts a Score record into the database, resolving goal_id and hypothesis_id.
        """
        try:
            with self.db.cursor() as cur:
                # Look up goal_id
                cur.execute("SELECT id FROM goals WHERE goal_text = %s LIMIT 1", (score.goal,))
                goal_row = cur.fetchone()
                if not goal_row:
                    raise ValueError(f"Goal not found: {score.goal}")
                goal_id = goal_row[0]

                # Look up hypothesis_id
                cur.execute("SELECT id FROM hypotheses WHERE text = %s LIMIT 1", (score.hypothesis,))
                hyp_row = cur.fetchone()
                if not hyp_row:
                    raise ValueError(f"Hypothesis not found: {score.hypothesis}")
                hypothesis_id = hyp_row[0]

                # Insert score
                cur.execute("""
                    INSERT INTO scores (
                        goal_id, hypothesis_id, agent_name, model_name, evaluator_name,
                        score_type, score, score_text, strategy, reasoning_strategy,
                        rationale, reflection, review, meta_review, run_id, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    goal_id,
                    hypothesis_id,
                    score.agent_name,
                    score.model_name,
                    score.evaluator_name,
                    score.score_type,
                    score.score,
                    score.score_text,
                    score.strategy,
                    score.reasoning_strategy,
                    score.rationale,
                    score.reflection,
                    score.review,
                    score.meta_review,
                    score.run_id,
                    json.dumps(score.metadata) or {}
                ))
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log(
                    "StoreScoreFailed",
                    {**score, **{"error": str(e)}}
                )

    def get_by_goal_id(self, goal_id: int) -> List[Dict]:
        """
        Returns all scores associated with a specific goal.
        """
        query = "SELECT * FROM scores WHERE goal_id = %s ORDER BY created_at DESC"
        return self._execute_and_fetch(query, (goal_id,))

    def get_by_run_id(self, run_id: str) -> list[Dict]:
        """
        Returns all scores associated with a specific goal.
        """
        query = "SELECT * FROM scores WHERE run_id = %s ORDER BY created_at DESC"
        return self._execute_and_fetch(query, (run_id,))

    def get_by_hypothesis_id(self, hypothesis_id: int) -> list[Dict]:
        """
        Returns all scores associated with a specific hypothesis.
        """
        query = "SELECT * FROM scores WHERE hypothesis_id = %s ORDER BY created_at DESC"
        return self._execute_and_fetch(query, (hypothesis_id,))

    def get_by_evaluator(self, evaluator_name: str) -> list[Dict]:
        """
        Returns all scores produced by a specific evaluator (e.g., 'llm', 'mrq').
        """
        query = "SELECT * FROM scores WHERE evaluator_name = %s ORDER BY created_at DESC"
        return self._execute_and_fetch(query, (evaluator_name,))

    def get_by_strategy(self, strategy: str) -> list[Dict]:
        """
        Returns all scores generated using a specific reasoning strategy.
        """
        query = "SELECT * FROM scores WHERE strategy = %s ORDER BY created_at DESC"
        return self._execute_and_fetch(query, (strategy,))

    def get_all(self, limit: int = 100) -> list[Dict]:
        """
        Returns the most recent scores up to a limit.
        """
        query = f"SELECT * FROM scores ORDER BY created_at DESC LIMIT {limit}"
        return self._execute_and_fetch(query)

    def _execute_and_fetch(self, query: str, params: tuple = None) -> list[Dict]:
        """
        Helper method to execute a query and fetch results as list of dicts.
        """
        try:
            with self.db.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                result = [dict(zip(columns, row)) for row in rows]
                return result
        except Exception as e:
            if self.logger:
                self.logger.log("ScoreFetchFailed", {"error": str(e)})
            return []