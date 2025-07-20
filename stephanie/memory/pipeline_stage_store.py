# stephanie/memory/pipeline_stage_store.py
from typing import List, Optional

from sqlalchemy.orm import Session

from stephanie.models.pipeline_stage import PipelineStageORM


class PipelineStageStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "pipeline_stages"

    def insert(self, stage_dict: dict) -> int:
        """
        Inserts a new pipeline stage into the database.

        :param stage_dict: Dictionary containing all required stage fields
        :return: The inserted stage's ID
        """
        try:
            # Convert dictionary to ORM object
            db_stage = PipelineStageORM(
                stage_name=stage_dict.get("stage_name"),
                agent_class=stage_dict.get("agent_class"),
                protocol_used=stage_dict.get("protocol_used"),
                goal_id=stage_dict.get("goal_id"),
                run_id=stage_dict.get("run_id"),
                pipeline_run_id=stage_dict.get("pipeline_run_id"),
                parent_stage_id=stage_dict.get("parent_stage_id"),
                input_context_id=stage_dict.get("input_context_id"),
                output_context_id=stage_dict.get("output_context_id"),
                status=stage_dict.get("status"),
                score=stage_dict.get("score"),
                confidence=stage_dict.get("confidence"),
                symbols_applied=stage_dict.get("symbols_applied"),
                extra_data=stage_dict.get("extra_data"),
                exportable=stage_dict.get("exportable"),
                reusable=stage_dict.get("reusable"),
                invalidated=stage_dict.get("invalidated"),
            )

            self.session.add(db_stage)
            self.session.flush()  # Get ID before commit
            stage_id = db_stage.id

            if self.logger:
                self.logger.log(
                    "PipelineStageInserted",
                    {
                        "stage_id": stage_id,
                        "stage_name": db_stage.stage_name,
                        "run_id": db_stage.run_id,
                        "status": db_stage.status,
                        "timestamp": db_stage.timestamp.isoformat(),
                    },
                )

            return stage_id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("PipelineStageInsertFailed", {"error": str(e)})
            raise

    def get_by_id(self, stage_id: int) -> Optional[PipelineStageORM]:
        """
        Fetches a pipeline stage by its database ID.

        :param stage_id: The ID of the stage to fetch
        :return: PipelineStageORM object or None
        """
        result = (
            self.session.query(PipelineStageORM)
            .filter(PipelineStageORM.id == stage_id)
            .first()
        )
        return result

    def get_by_run_id(self, run_id: str) -> List[PipelineStageORM]:
        """
        Fetches all stages for a given run_id.

        :param run_id: Unique ID of the pipeline run
        :return: List of PipelineStageORM objects
        """
        return (
            self.session.query(PipelineStageORM)
            .filter(PipelineStageORM.run_id == run_id)
            .order_by(PipelineStageORM.timestamp.asc())
            .all()
        )

    def get_by_pipeline_run_id(self, pipeline_run_id: int) -> List[PipelineStageORM]:
        """
        Fetches all stages associated with a given pipeline_run.

        :param pipeline_run_id: ID of the pipeline run
        :return: List of PipelineStageORM objects
        """
        return (
            self.session.query(PipelineStageORM)
            .filter(PipelineStageORM.pipeline_run_id == pipeline_run_id)
            .order_by(PipelineStageORM.timestamp.asc())
            .all()
        )

    def get_by_goal_id(self, goal_id: int) -> List[PipelineStageORM]:
        """
        Fetches all stages associated with a given goal.

        :param goal_id: ID of the goal
        :return: List of PipelineStageORM objects
        """
        return (
            self.session.query(PipelineStageORM)
            .filter(PipelineStageORM.goal_id == goal_id)
            .order_by(PipelineStageORM.timestamp.asc())
            .all()
        )

    def get_by_parent_stage_id(self, parent_stage_id: int) -> List[PipelineStageORM]:
        """
        Fetches all child stages of a given stage.

        :param parent_stage_id: ID of the parent stage
        :return: List of PipelineStageORM objects
        """
        return (
            self.session.query(PipelineStageORM)
            .filter(PipelineStageORM.parent_stage_id == parent_stage_id)
            .order_by(PipelineStageORM.timestamp.asc())
            .all()
        )

    def get_all(self, limit: int = 100) -> List[PipelineStageORM]:
        """
        Returns the most recent pipeline stages up to a limit.

        :param limit: Maximum number of stages to return
        :return: List of PipelineStageORM objects
        """
        return (
            self.session.query(PipelineStageORM)
            .order_by(PipelineStageORM.timestamp.desc())
            .limit(limit)
            .all()
        )

    def find(self, filters: dict) -> List[PipelineStageORM]:
        """
        Generic search method for pipeline stages.

        :param filters: Dictionary of filter conditions
        :return: Matching PipelineStageORM instances
        """
        query = self.session.query(PipelineStageORM)

        if "run_id" in filters:
            query = query.filter(PipelineStageORM.run_id == filters["run_id"])
        if "stage_name" in filters:
            query = query.filter(PipelineStageORM.stage_name == filters["stage_name"])
        if "protocol_used" in filters:
            query = query.filter(PipelineStageORM.protocol_used == filters["protocol_used"])
        if "status" in filters:
            query = query.filter(PipelineStageORM.status == filters["status"])
        if "goal_id" in filters:
            query = query.filter(PipelineStageORM.goal_id == filters["goal_id"])
        if "pipeline_run_id" in filters:
            query = query.filter(PipelineStageORM.pipeline_run_id == filters["pipeline_run_id"])
        if "since" in filters:
            query = query.filter(PipelineStageORM.timestamp >= filters["since"])

        return query.order_by(PipelineStageORM.timestamp.desc()).all()
    

    def get_reasoning_tree(self, run_id: str) -> list[dict[str, any]]:
        """
        Builds a recursive reasoning tree of all stages for a given run_id.

        :param run_id: The run ID to build the tree for.
        :return: A list of root nodes with full nested children.
        """
        # Fetch all stages for this run
        stages = (
            self.session.query(PipelineStageORM)
            .filter(PipelineStageORM.run_id == run_id)
            .all()
        )

        if not stages:
            return []

        # Map stages by ID for quick lookup
        stage_map = {stage.id: self._stage_to_dict(stage) for stage in stages}

        # Build tree structure
        tree = []
        for stage in stages:
            stage_dict = stage_map[stage.id]
            parent_id = stage.parent_stage_id

            if parent_id is None:
                tree.append(stage_dict)
            else:
                if parent_id in stage_map:
                    stage_map[parent_id].setdefault("children", []).append(stage_dict)
                else:
                    # Orphaned child — log or handle accordingly
                    tree.append(stage_dict)

        return tree

    def _stage_to_dict(self, stage: PipelineStageORM) -> dict[str, any]:
        """
        Converts a PipelineStageORM object into a dictionary.
        """
        return {
            "id": stage.id,
            "stage_name": stage.stage_name,
            "agent_class": stage.agent_class,
            "protocol_used": stage.protocol_used,
            "status": stage.status,
            "score": stage.score,
            "timestamp": stage.timestamp.isoformat(),
            "input_context_id": stage.input_context_id,
            "output_context_id": stage.output_context_id,
            "children": []
        }