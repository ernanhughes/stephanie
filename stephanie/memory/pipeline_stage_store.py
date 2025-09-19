# stephanie/memory/pipeline_stage_store.py
from __future__ import annotations

from typing import List, Optional

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.pipeline_stage import PipelineStageORM


class PipelineStageStore(BaseSQLAlchemyStore):
    orm_model = PipelineStageORM
    default_order_by = PipelineStageORM.timestamp.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "pipeline_stages"

    def name(self) -> str:
        return self.name

    def insert(self, stage_dict: dict) -> int:
        """
        Insert a new pipeline stage into the database.

        :param stage_dict: Dictionary containing all required stage fields
        :return: The inserted stage's ID
        """
        def op():
            s = self._scope()
            db_stage = PipelineStageORM(**stage_dict)
            s.add(db_stage)
            s.flush()  # Assign ID
            if self.logger:
                self.logger.log(
                    "PipelineStageInserted",
                    {
                        "stage_id": db_stage.id,
                        "stage_name": db_stage.stage_name,
                        "run_id": db_stage.run_id,
                        "status": db_stage.status,
                        "timestamp": db_stage.timestamp.isoformat()
                        if db_stage.timestamp else None,
                    },
                )
            return db_stage.id
        return self._run(op)

    def get_by_id(self, stage_id: int) -> Optional[PipelineStageORM]:
        def op():
            return (
                self._scope()
                .query(PipelineStageORM)
                .filter(PipelineStageORM.id == stage_id)
                .first()
            )
        return self._run(op)

    def get_by_run_id(self, run_id: str) -> List[PipelineStageORM]:
        def op():
            return (
                self._scope()
                .query(PipelineStageORM)
                .filter(PipelineStageORM.pipeline_run_id == run_id)
                .order_by(PipelineStageORM.timestamp.asc())
                .all()
            )
        return self._run(op)

    def get_by_goal_id(self, goal_id: int) -> List[PipelineStageORM]:
        def op():
            return (
                self._scope()
                .query(PipelineStageORM)
                .filter(PipelineStageORM.goal_id == goal_id)
                .order_by(PipelineStageORM.timestamp.asc())
                .all()
            )
        return self._run(op)

    def get_by_parent_stage_id(self, parent_stage_id: int) -> List[PipelineStageORM]:
        def op():
            return (
                self._scope()
                .query(PipelineStageORM)
                .filter(PipelineStageORM.parent_stage_id == parent_stage_id)
                .order_by(PipelineStageORM.timestamp.asc())
                .all()
            )
        return self._run(op)

    def get_all(self, limit: int = 100) -> List[PipelineStageORM]:
        def op():
            return (
                self._scope()
                .query(PipelineStageORM)
                .order_by(PipelineStageORM.timestamp.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def find(self, filters: dict) -> List[PipelineStageORM]:
        def op():
            q = self._scope().query(PipelineStageORM)
            if "run_id" in filters:
                q = q.filter(PipelineStageORM.run_id == filters["run_id"])
            if "stage_name" in filters:
                q = q.filter(PipelineStageORM.stage_name == filters["stage_name"])
            if "status" in filters:
                q = q.filter(PipelineStageORM.status == filters["status"])
            if "goal_id" in filters:
                q = q.filter(PipelineStageORM.goal_id == filters["goal_id"])
            if "pipeline_run_id" in filters:
                q = q.filter(PipelineStageORM.pipeline_run_id == filters["pipeline_run_id"])
            if "since" in filters:
                q = q.filter(PipelineStageORM.timestamp >= filters["since"])
            return q.order_by(PipelineStageORM.timestamp.desc()).all()
        return self._run(op)

    def get_reasoning_tree(self, run_id: str) -> list[dict[str, any]]:
        """
        Build a recursive reasoning tree of all stages for a given run_id.
        """
        def op():
            stages = (
                self._scope()
                .query(PipelineStageORM)
                .filter(PipelineStageORM.run_id == run_id)
                .all()
            )
            if not stages:
                return []
            stage_map = {s.id: self._stage_to_dict(s) for s in stages}
            tree = []
            for s in stages:
                s_dict = stage_map[s.id]
                if s.parent_stage_id is None:
                    tree.append(s_dict)
                elif s.parent_stage_id in stage_map:
                    stage_map[s.parent_stage_id].setdefault("children", []).append(s_dict)
                else:
                    tree.append(s_dict)  # orphan
            return tree
        return self._run(op)

    def _stage_to_dict(self, stage: PipelineStageORM) -> dict[str, any]:
        return {
            "id": stage.id,
            "stage_name": stage.stage_name,
            "agent_class": stage.agent_class,
            "status": stage.status,
            "score": stage.score,
            "timestamp": stage.timestamp.isoformat() if stage.timestamp else None,
            "input_context_id": stage.input_context_id,
            "output_context_id": stage.output_context_id,
            "children": [],
        }
