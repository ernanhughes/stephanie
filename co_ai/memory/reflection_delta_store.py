import json
from typing import Optional, List

from co_ai.memory.base_store import BaseStore
from co_ai.models.reflection_delta import ReflectionDelta


class ReflectionDeltaStore(BaseStore):
    def __init__(self, db, logger=None):
        super().__init__(db, logger)
        self.name = "reflection_deltas"

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "reflection_deltas"

    def insert(self, delta: ReflectionDelta) -> int:
        """
        Inserts a new reflection delta into the database.

        :param delta: A ReflectionDelta dataclass instance
        :return: The inserted record's ID
        """
        query = """
            INSERT INTO reflection_deltas (
                goal_id,
                run_id_a,
                run_id_b,
                score_a,
                score_b,
                score_delta,
                pipeline_a,
                pipeline_b,
                pipeline_diff,
                strategy_diff,
                model_diff,
                rationale_diff,
                created_at
            ) VALUES (
                %(goal_id)s,
                %(run_id_a)s,
                %(run_id_b)s,
                %(score_a)s,
                %(score_b)s,
                %(score_delta)s,
                %(pipeline_a)s,
                %(pipeline_b)s,
                %(pipeline_diff)s,
                %(strategy_diff)s,
                %(model_diff)s,
                %(rationale_diff)s,
                %(created_at)s
            )
            RETURNING id;
        """

        try:
            with self.db.cursor() as cur:
                cur.execute(query, {
                    "goal_id": delta.goal_id,
                    "run_id_a": delta.run_id_a,
                    "run_id_b": delta.run_id_b,
                    "score_a": delta.score_a,
                    "score_b": delta.score_b,
                    "score_delta": delta.score_delta,
                    "pipeline_a": json.dumps(delta.pipeline_a or {}),
                    "pipeline_b": json.dumps(delta.pipeline_b or {}),
                    "pipeline_diff": json.dumps(delta.pipeline_diff or {}),
                    "strategy_diff": delta.strategy_diff or False,
                    "model_diff": delta.model_diff or False,
                    "rationale_diff": json.dumps(list(delta.rationale_diff)) if delta.rationale_diff else json.dumps(["", ""]),
                    "created_at": delta.created_at
                })
                delta_id = cur.fetchone()[0]
                self.db.commit()

                if self.logger:
                    self.logger.log("ReflectionDeltaInserted", {
                        "delta_id": delta_id,
                        "goal_id": delta.goal_id,
                        "run_id_a": delta.run_id_a,
                        "run_id_b": delta.run_id_b,
                        "score_a": delta.score_a,
                        "score_b": delta.score_b,
                        "score_delta": delta.score_delta,
                        "strategy_diff": delta.strategy_diff,
                        "model_diff": delta.model_diff,
                    })

                return delta_id

        except Exception as e:
            self.db.rollback()
            if self.logger:
                self.logger.log("ReflectionDeltaInsertFailed", {"error": str(e)})
            raise

    def get_by_goal_id(self, goal_id: int) -> list[ReflectionDelta]:
        """
        Fetches all reflection deltas associated with a given goal.

        :param goal_id: Goal ID to filter by
        :return: List of ReflectionDelta objects
        """
        query = "SELECT * FROM reflection_deltas WHERE goal_id = %s ORDER BY created_at DESC"
        return self._fetch_all(query, (goal_id,))

    def get_by_run_ids(self, run_id_a: str, run_id_b: str) -> Optional[ReflectionDelta]:
        """
        Fetches a reflection delta comparing two specific runs.

        :param run_id_a: First run ID
        :param run_id_b: Second run ID
        :return: ReflectionDelta object or None
        """
        query = "SELECT * FROM reflection_deltas WHERE run_id_a = %s AND run_id_b = %s"
        return self._fetch_one(query, (run_id_a, run_id_b))

    def get_all(self, limit: int = 100) -> list[ReflectionDelta]:
        """
        Returns the most recent reflection deltas up to a limit.

        :param limit: Max number of results to return
        :return: List of ReflectionDelta objects
        """
        query = f"SELECT * FROM reflection_deltas ORDER BY created_at DESC LIMIT {limit}"
        return self._fetch_all(query)

    def find(self, filters: dict) -> list[ReflectionDelta]:
        """
        Generic search method for reflection deltas.

        :param filters: Dictionary of filter conditions
        :return: Matching ReflectionDelta instances
        """
        where_clauses = []
        params = {}

        if "goal_id" in filters:
            where_clauses.append("goal_id = %(goal_id)s")
            params["goal_id"] = filters["goal_id"]

        if "run_id_a" in filters and "run_id_b" in filters:
            where_clauses.append("run_id_a = %(run_id_a)s AND run_id_b = %(run_id_b)s")
            params["run_id_a"] = filters["run_id_a"]
            params["run_id_b"] = filters["run_id_b"]

        if "score_delta_gt" in filters:
            where_clauses.append("score_delta > %(score_delta_gt)s")
            params["score_delta_gt"] = filters["score_delta_gt"]

        if "strategy_diff" in filters:
            where_clauses.append("strategy_diff = %(strategy_diff)s")
            params["strategy_diff"] = filters["strategy_diff"]

        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        query = f"SELECT * FROM reflection_deltas {where_clause} ORDER BY created_at DESC"

        return self._fetch_all(query, params)

    def _fetch_one(self, query: str, params=None) -> Optional[ReflectionDelta]:
        """
        Helper to fetch one result and convert it to a ReflectionDelta object.
        """
        try:
            with self.db.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_to_delta(row)
        except Exception as e:
            if self.logger:
                self.logger.log("ReflectionDeltaFetchFailed", {"error": str(e)})
            return None

    def _fetch_all(self, query: str, params=None) -> List[ReflectionDelta]:
        """
        Helper to fetch multiple results and convert them to ReflectionDelta objects.
        """
        try:
            with self.db.cursor() as cur:
                cur.execute(query, params or {})
                rows = cur.fetchall()
                return [self._row_to_delta(row) for row in rows]
        except Exception as e:
            if self.logger:
                self.logger.log("ReflectionDeltasFetchFailed", {"error": str(e)})
            return []

    def _row_to_delta(self, row) -> ReflectionDelta:
        """
        Converts a database row to a ReflectionDelta object.
        """
        return ReflectionDelta(
            id=row["id"],
            goal_id=row["goal_id"],
            run_id_a=row["run_id_a"],
            run_id_b=row["run_id_b"],
            score_a=row.get("score_a"),
            score_b=row.get("score_b"),
            score_delta=row.get("score_delta"),
            pipeline_a=json.loads(row.get("pipeline_a")) if row.get("pipeline_a") else {},
            pipeline_b=json.loads(row.get("pipeline_b")) if row.get("pipeline_b") else {},
            pipeline_diff=json.loads(row.get("pipeline_diff")) if row.get("pipeline_diff") else {},
            strategy_diff=row.get("strategy_diff"),
            model_diff=row.get("model_diff"),
            rationale_diff=tuple(json.loads(row.get("rationale_diff"))) if row.get("rationale_diff") else ("", ""),
            created_at=row.get("created_at")
        )