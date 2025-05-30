# stores/pipeline_run_store.py
import json
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from co_ai.models.pipeline_run import PipelineRunORM


class PipelineRunStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "pipeline_runs"

    def insert(self, run_dict: dict) -> int:
        """
        Inserts a new pipeline run record into the database.
        
        :param run_dict: Dictionary containing fields like run_id, goal_id, pipeline, etc.
        :return: The inserted record's ID
        """
        try:
            # Convert dictionary to ORM object
            db_run = PipelineRunORM(
                run_id=run_dict["run_id"],
                # goal_id=run_dict["goal_id"],
                pipeline=run_dict.get("pipeline"),
                strategy=run_dict.get("strategy"),
                model_name=run_dict.get("model_name"),
                run_config=json.dumps(run_dict.get("run_config")) if run_dict.get("run_config") else None,
                lookahead_context=json.dumps(run_dict.get("lookahead_context")) if run_dict.get("lookahead_context") else None,
                symbolic_suggestion=json.dumps(run_dict.get("symbolic_suggestion")) if run_dict.get("symbolic_suggestion") else None,
                extra_data=json.dumps(run_dict.get("extra_data")) if run_dict.get("extra_data") else None,
                created_at=run_dict.get("created_at") or datetime.utcnow()
            )

            self.session.add(db_run)
            self.session.flush()  # Get ID before commit
            run_id = db_run.id

            if self.logger:
                self.logger.log("PipelineRunInserted", {
                    "run_id": db_run.run_id,
                    "goal_id": db_run.goal_id,
                    "pipeline": db_run.pipeline,
                    "strategy": db_run.strategy,
                    "model": db_run.model_name,
                    "timestamp": db_run.created_at if db_run.created_at else None
                })

            return run_id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("PipelineRunInsertFailed", {"error": str(e)})
            raise

    def get_by_run_id(self, run_id: str) -> Optional[PipelineRunORM]:
        """
        Fetches a single pipeline run by its unique run_id.
        """
        result = self.session.query(PipelineRunORM).filter(PipelineRunORM.run_id == run_id).first()
        return result

    def get_by_goal_id(self, goal_id: int) -> list[PipelineRunORM]:
        """
        Fetches all pipeline runs associated with a given goal.
        """
        return self.session.query(PipelineRunORM).filter(PipelineRunORM.goal_id == goal_id).all()

    def get_all(self, limit: int = 100) -> list[PipelineRunORM]:
        """
        Returns the most recent pipeline runs up to a limit.
        """
        return self.session.query(PipelineRunORM).order_by(PipelineRunORM.created_at.desc()).limit(limit).all()

    def find(self, filters: dict) -> list[PipelineRunORM]:
        """
        Generic search method for pipeline runs.

        :param filters: Dictionary of filter conditions
        :return: Matching PipelineRun instances
        """
        query = self.session.query(PipelineRunORM)

        if "goal_id" in filters:
            query = query.filter(PipelineRunORM.goal_id == filters["goal_id"])
        if "strategy" in filters:
            query = query.filter(PipelineRunORM.strategy == filters["strategy"])
        if "model_name" in filters:
            query = query.filter(PipelineRunORM.model_name == filters["model_name"])
        if "since" in filters:
            query = query.filter(PipelineRunORM.created_at >= filters["since"])

        return query.order_by(PipelineRunORM.created_at.desc()).all()