# stephanie/memory/pipeline_reference_store.py
from __future__ import annotations

import logging
from typing import List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.pipeline_reference import PipelineReferenceORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable import ScorableFactory

logger = logging.getLogger(__name__)


class PipelineReferenceStore(BaseSQLAlchemyStore):
    orm_model = PipelineReferenceORM
    default_order_by = PipelineReferenceORM.created_at.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "pipeline_references"

    def name(self) -> str:
        return self.name

    def insert(self, reference_dict: dict) -> int:
        """
        Inserts a new pipeline reference into the database.
        """
        def op(s):
            db_ref = PipelineReferenceORM(
                pipeline_run_id=reference_dict.get("pipeline_run_id"),
                scorable_type=reference_dict.get("scorable_type"),
                scorable_id=reference_dict.get("scorable_id"),
                relation_type=reference_dict.get("relation_type"),
                source=reference_dict.get("source"),
            )
            s.add(db_ref)
            s.flush()  # get ID
            if self.logger:
                self.logger.log(
                    "PipelineReferenceInserted",
                    {
                        "reference_id": db_ref.id,
                        "pipeline_run_id": db_ref.pipeline_run_id,
                        "scorable_type": db_ref.scorable_type,
                        "scorable_id": db_ref.scorable_id,
                    },
                )
            return db_ref.id
        return self._run(op)

    def get_by_id(self, ref_id: int) -> Optional[PipelineReferenceORM]:
        def op(s):
            return self._scope().get(PipelineReferenceORM, ref_id)
        return self._run(op)

    def get_by_pipeline_run(self, pipeline_run_id: int) -> List[PipelineReferenceORM]:
        def op(s):
            return (
                self._scope()
                .query(PipelineReferenceORM)
                .filter(PipelineReferenceORM.pipeline_run_id == pipeline_run_id)
                .order_by(PipelineReferenceORM.created_at.asc())
                .all()
            )
        return self._run(op)

    def get_by_target(self, target_type: str, target_id: str) -> List[PipelineReferenceORM]:
        def op(s):
            return (
                self._scope()
                .query(PipelineReferenceORM)
                .filter(
                    PipelineReferenceORM.scorable_type == target_type,
                    PipelineReferenceORM.scorable_id == target_id,
                )
                .order_by(PipelineReferenceORM.created_at.asc())
                .all()
            )
        return self._run(op)

    def get_all(self, limit: int = 100) -> List[PipelineReferenceORM]:
        def op(s):
            return (
                self._scope()
                .query(PipelineReferenceORM)
                .order_by(PipelineReferenceORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def find(self, filters: dict) -> List[PipelineReferenceORM]:
        def op(s):
            q = s.query(PipelineReferenceORM)
            if "pipeline_run_id" in filters:
                q = q.filter(PipelineReferenceORM.pipeline_run_id == filters["pipeline_run_id"])
            if "scorable_type" in filters:
                q = q.filter(PipelineReferenceORM.scorable_type == filters["scorable_type"])
            if "scorable_id" in filters:
                q = q.filter(PipelineReferenceORM.scorable_id == filters["scorable_id"])
            if "relation_type" in filters:
                q = q.filter(PipelineReferenceORM.relation_type == filters["relation_type"])
            if "since" in filters:
                q = q.filter(PipelineReferenceORM.created_at >= filters["since"])
            return q.order_by(PipelineReferenceORM.created_at.desc()).all()
        return self._run(op)

    def get_documents_by_run_id(
        self, pipeline_run_id: int, memory, limit: int = 100
    ) -> dict[tuple[str, str], Scorable]:
        def op(s):
            refs = (
                self._scope()
                .query(PipelineReferenceORM)
                .filter(PipelineReferenceORM.pipeline_run_id == pipeline_run_id)
                .order_by(PipelineReferenceORM.created_at.asc())
                .limit(limit)
                .all()
            )

            results: dict[tuple[str, str], Scorable] = {}
            for ref in refs:
                try:
                    scorable = ScorableFactory.from_id(
                        memory, ref.scorable_type, ref.scorable_id
                    )
                    results[(ref.scorable_type, ref.scorable_id)] = scorable
                except Exception as e:
                    if self.logger:
                        self.logger.log(
                            "PipelineReferenceResolveFailed",
                            {
                                "pipeline_run_id": pipeline_run_id,
                                "scorable_type": ref.scorable_type,
                                "scorable_id": ref.scorable_id,
                                "error": str(e),
                            },
                        )
            return results
        return self._run(op)
