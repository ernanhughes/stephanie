from typing import Optional, List, Dict
from datetime import datetime

from co_ai.memory.base_store import BaseStore
from co_ai.models.pipeline_run import PipelineRun
import json

class PipelineRunStore(BaseStore):
    def __init__(self, db, logger=None):
        super().__init__(db, logger)
        self.name = "pipeline_runs"

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "pipeline_runs"

    def insert(self, run: PipelineRun) -> int:
        """
        Inserts a new pipeline run record into the database.

        :param run: A PipelineRun dataclass instance
        :return: The inserted record's ID
        """
        query = """
            INSERT INTO pipeline_runs (
                run_id,
                goal_id,
                pipeline,
                strategy,
                model_name,
                run_config,
                lookahead_context,
                symbolic_suggestion,
                metadata,
                created_at
            ) VALUES (
                %(run_id)s,
                %(goal_id)s,
                %(pipeline)s,
                %(strategy)s,
                %(model_name)s,
                %(run_config)s,
                %(lookahead_context)s,
                %(symbolic_suggestion)s,
                %(metadata)s,
                %(created_at)s
            )
            RETURNING id;
        """

        try:
            with self.db.cursor() as cur:
                cur.execute(query, {
                    "run_id": run.run_id,
                    "goal_id": run.goal_id,
                    "pipeline": json.dumps(run.pipeline) or {},
                    "strategy": run.strategy,
                    "model_name": run.model_name,
                    "run_config": json.dumps(run.run_config) or {},
                    "lookahead_context": json.dumps(run.lookahead_context) or {},
                    "symbolic_suggestion": json.dumps(run.symbolic_suggestion) or {},
                    "metadata": json.dumps(run.metadata) or {},
                    "created_at": run.created_at
                })
                run_id = cur.fetchone()[0]
                self.db.commit()

                if self.logger:
                    self.logger.log("PipelineRunInserted", {
                        "run_id": run.run_id,
                        "goal_id": run.goal_id,
                        "pipeline": run.pipeline,
                        "strategy": run.strategy,
                        "model": run.model_name,
                        "timestamp": run.created_at.isoformat() if run.created_at else None
                    })

                return run_id

        except Exception as e:
            self.db.rollback()
            if self.logger:
                self.logger.log("PipelineRunInsertFailed", {"error": str(e)})
            raise

    def get_by_run_id(self, run_id: str) -> Optional[PipelineRun]:
        """
        Fetches a single pipeline run by its unique run_id.

        :param run_id: Unique identifier for the run
        :return: PipelineRun object or None
        """
        query = "SELECT * FROM pipeline_runs WHERE run_id = %s"

        return self._fetch_one(query, (run_id,))

    def get_by_goal_id(self, goal_id: int) -> list[PipelineRun]:
        """
        Fetches all pipeline runs associated with a given goal.

        :param goal_id: Goal ID to filter by
        :return: List of PipelineRun objects
        """
        query = "SELECT * FROM pipeline_runs WHERE goal_id = %s ORDER BY created_at DESC"
        return self._fetch_all(query, (goal_id,))

    def get_all(self, limit: int = 100) -> list[PipelineRun]:
        """
        Returns the most recent pipeline runs up to a limit.

        :param limit: Max number of results to return
        :return: List of PipelineRun objects
        """
        query = f"SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT {limit}"
        return self._fetch_all(query)

    def find(self, filters: dict) -> list[PipelineRun]:
        """
        Generic search method for pipeline runs.

        :param filters: Dictionary of filter conditions
        :return: Matching PipelineRun instances
        """
        where_clauses = []
        params = {}

        if "goal_id" in filters:
            where_clauses.append("goal_id = %(goal_id)s")
            params["goal_id"] = filters["goal_id"]

        if "strategy" in filters:
            where_clauses.append("strategy = %(strategy)s")
            params["strategy"] = filters["strategy"]

        if "model_name" in filters:
            where_clauses.append("model_name = %(model_name)s")
            params["model_name"] = filters["model_name"]

        if "since" in filters:
            where_clauses.append("created_at >= %(since)s")
            params["since"] = filters["since"]

        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        query = f"SELECT * FROM pipeline_runs {where_clause} ORDER BY created_at DESC"

        return self._fetch_all(query, params)

    def _fetch_one(self, query: str, params=None) -> Optional[PipelineRun]:
        """
        Helper to fetch one result and convert it to a PipelineRun object.
        """
        try:
            with self.db.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_to_run(row)
        except Exception as e:
            if self.logger:
                self.logger.log("PipelineRunFetchFailed", {"error": str(e)})
            return None

    def _fetch_all(self, query: str, params=None) -> list[PipelineRun]:
        """
        Helper to fetch multiple results and convert them to PipelineRun objects.
        """
        try:
            with self.db.cursor() as cur:
                cur.execute(query, params or {})
                rows = cur.fetchall()
                return [self._row_to_run(row) for row in rows]
        except Exception as e:
            if self.logger:
                self.logger.log("PipelineRunsFetchFailed", {"error": str(e)})
            return []

    def _row_to_run(self, row) -> PipelineRun:
        """
        Converts a database row to a PipelineRun object.
        """
        return PipelineRun(
            id=row["id"],
            run_id=row["run_id"],
            goal_id=row["goal_id"],
            pipeline=row["pipeline"],
            strategy=row.get("strategy"),
            model_name=row.get("model_name"),
            run_config=row.get("run_config"),
            lookahead_context=row.get("lookahead_context"),
            symbolic_suggestion=row.get("symbolic_suggestion"),
            metadata=row.get("metadata"),
            created_at=row.get("created_at")
        )