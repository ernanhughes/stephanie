# stephanie/memory/pipeline_reference_store.py
from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.pipeline_reference import PipelineReferenceORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory

logger = logging.getLogger(__name__)


class PipelineReferenceStore(BaseSQLAlchemyStore):
    orm_model = PipelineReferenceORM
    default_order_by = PipelineReferenceORM.created_at.desc()
    
    def __init__(self, session: Session, logger=None):
        super().__init__(session, logger)
        self.name = "pipeline_references"

    def name(self) -> str:
        return self.name

    def insert(self, reference_dict: dict) -> int:
        """
        Inserts a new pipeline reference into the database.

        :param reference_dict: Dictionary with pipeline_run_id, target_type, target_id, relation_type
        :return: The inserted reference's ID
        """
        try:
            db_ref = PipelineReferenceORM(
                pipeline_run_id=reference_dict.get("pipeline_run_id"),
                scorable_type=reference_dict.get("scorable_type"),
                scorable_id=reference_dict.get("scorable_id"),
                relation_type=reference_dict.get("relation_type"),
                source=reference_dict.get("source"),
            )

            self.session.add(db_ref)
            self.session.flush()  # Get ID before commit
            ref_id = db_ref.id

            logger.debug(
                "PipelineReferenceInserted"
                f"reference_id: {ref_id}, "
                f"pipeline_run_id: {db_ref.pipeline_run_id}, "
                f"scorable_type: {db_ref.scorable_type}, "
                f"scorable_id: {db_ref.scorable_id}"
            )

            self.session.commit()
            return ref_id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log(
                    "PipelineReferenceInsertFailed", {"error": str(e)}
                )
            raise

    def get_by_id(self, ref_id: int) -> Optional[PipelineReferenceORM]:
        """
        Fetch a pipeline reference by ID.
        """
        return (
            self.session.query(PipelineReferenceORM)
            .filter(PipelineReferenceORM.id == ref_id)
            .first()
        )

    def get_by_pipeline_run(
        self, pipeline_run_id: int
    ) -> List[PipelineReferenceORM]:
        """
        Fetch all references associated with a pipeline run.
        """
        return (
            self.session.query(PipelineReferenceORM)
            .filter(PipelineReferenceORM.pipeline_run_id == pipeline_run_id)
            .order_by(PipelineReferenceORM.created_at.asc())
            .all()
        )

    def get_by_target(
        self, target_type: str, target_id: str
    ) -> List[PipelineReferenceORM]:
        """
        Fetch all references to a specific scorable target.
        """
        return (
            self.session.query(PipelineReferenceORM)
            .filter(PipelineReferenceORM.scorable_type == target_type)
            .filter(PipelineReferenceORM.scorable_id == target_id)
            .order_by(PipelineReferenceORM.created_at.asc())
            .all()
        )

    def get_all(self, limit: int = 100) -> List[PipelineReferenceORM]:
        """
        Return the most recent references.
        """
        return (
            self.session.query(PipelineReferenceORM)
            .order_by(PipelineReferenceORM.created_at.desc())
            .limit(limit)
            .all()
        )

    def find(self, filters: dict) -> List[PipelineReferenceORM]:
        """
        Generic search method for pipeline references.
        """
        query = self.session.query(PipelineReferenceORM)

        if "pipeline_run_id" in filters:
            query = query.filter(
                PipelineReferenceORM.pipeline_run_id
                == filters["pipeline_run_id"]
            )
        if "scorable_type" in filters:
            query = query.filter(
                PipelineReferenceORM.scorable_type == filters["scorable_type"]
            )
        if "scorable_id" in filters:
            query = query.filter(
                PipelineReferenceORM.scorable_id == filters["scorable_id"]
            )
        if "relation_type" in filters:
            query = query.filter(
                PipelineReferenceORM.relation_type == filters["relation_type"]
            )
        if "since" in filters:
            query = query.filter(
                PipelineReferenceORM.created_at >= filters["since"]
            )

        return query.order_by(PipelineReferenceORM.created_at.desc()).all()

    def get_documents_by_run_id(
        self, pipeline_run_id: int, memory, limit: int = 100
    ) -> dict:
        """
        Fetch the first `limit` referenced documents for a given pipeline run,
        and return them as Scorable objects keyed by (target_type, target_id).

        :param pipeline_run_id: The pipeline run ID
        :param memory: The memory object to resolve ORM objects
        :param limit: Maximum number of references to return (default=100)
        :return: dict of {(target_type, target_id): Scorable}
        """
        refs = (
            self.session.query(PipelineReferenceORM)
            .filter(PipelineReferenceORM.pipeline_run_id == pipeline_run_id)
            .order_by(PipelineReferenceORM.created_at.asc())
            .limit(limit)
            .all()
        )

        results: dict[tuple[str, str], Scorable] = {}
        for ref in refs:
            try:
                scorable = ScorableFactory.from_id(
                    memory, ref.target_type, ref.target_id
                )
                results[(ref.target_type, ref.target_id)] = scorable
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
