# stephanie/memory/pipeline_run_store.py
from __future__ import annotations

from typing import Optional

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.pipeline_run import PipelineRunORM


class PipelineRunStore(BaseSQLAlchemyStore):
    orm_model = PipelineRunORM
    default_order_by = PipelineRunORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "pipeline_runs"

    def name(self) -> str:
        return self.name

    def insert(self, run_dict: dict) -> int:
        """
        Insert a new pipeline run record into the database.

        :param run_dict: Dictionary containing fields like run_id, goal_id, pipeline, etc.
        :return: The inserted record's ID
        """
        def op():
            s = self._scope()
            db_run = PipelineRunORM(**run_dict)
            s.add(db_run)
            s.flush()  # assign ID
            if self.logger:
                self.logger.log(
                    "PipelineRunInserted",
                    {
                        "id": db_run.id,
                        "run_id": db_run.run_id,
                        "goal_id": db_run.goal_id,
                        "pipeline": db_run.pipeline,
                        "strategy": db_run.strategy,
                        "model": db_run.model_name,
                        "timestamp": db_run.created_at.isoformat()
                        if db_run.created_at else None,
                    },
                )
            return db_run.id
        return self._run(op)

    def get_by_run_id(self, run_id: int) -> Optional[PipelineRunORM]:
        """
        Fetch a single pipeline run by its database ID.
        """
        def op():
            return (
                self._scope()
                .query(PipelineRunORM)
                .filter(PipelineRunORM.id == run_id)
                .order_by(PipelineRunORM.created_at.desc())
                .first()
            )
        return self._run(op)

    def get_by_goal_id(self, goal_id: int) -> list[PipelineRunORM]:
        """
        Fetch all pipeline runs associated with a given goal.
        """
        def op():
            return (
                self._scope()
                .query(PipelineRunORM)
                .filter(PipelineRunORM.goal_id == goal_id)
                .order_by(PipelineRunORM.created_at.desc())
                .all()
            )
        return self._run(op)

    def get_all(self, limit: int = 100) -> list[PipelineRunORM]:
        """
        Return the most recent pipeline runs up to a limit.
        """
        def op():
            return (
                self._scope()
                .query(PipelineRunORM)
                .order_by(PipelineRunORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def find(self, filters: dict) -> list[PipelineRunORM]:
        """
        Generic search method for pipeline runs.
        """
        def op():
            q = self._scope().query(PipelineRunORM)

            if "goal_id" in filters:
                q = q.filter(PipelineRunORM.goal_id == filters["goal_id"])
            if "name" in filters:
                q = q.filter(PipelineRunORM.name == filters["name"])
            if "tag" in filters:
                q = q.filter(PipelineRunORM.tag == filters["tag"])
            if "strategy" in filters:
                q = q.filter(PipelineRunORM.strategy == filters["strategy"])
            if "model_name" in filters:
                q = q.filter(PipelineRunORM.model_name == filters["model_name"])
            if "since" in filters:
                q = q.filter(PipelineRunORM.created_at >= filters["since"])

            return q.order_by(PipelineRunORM.created_at.desc()).all()
        return self._run(op)
