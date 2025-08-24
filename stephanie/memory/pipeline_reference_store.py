# stephanie/memory/pipeline_reference_store.py
from typing import List, Optional

from sqlalchemy.orm import Session

from stephanie.models.pipeline_reference import PipelineReferenceORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory


class PipelineReferenceStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "pipeline_references"

    def insert(self, reference_dict: dict) -> int:
        """
        Inserts a new pipeline reference into the database.

        :param reference_dict: Dictionary with pipeline_run_id, target_type, target_id, relation_type
        :return: The inserted reference's ID
        """
        try:
            db_ref = PipelineReferenceORM(
                pipeline_run_id=reference_dict.get("pipeline_run_id"),
                target_type=reference_dict.get("target_type"),
                target_id=reference_dict.get("target_id"),
                relation_type=reference_dict.get("relation_type"),
                source=reference_dict.get("source"),
            )

            self.session.add(db_ref)
            self.session.flush()  # Get ID before commit
            ref_id = db_ref.id

            if self.logger:
                self.logger.log(
                    "PipelineReferenceInserted",
                    {
                        "reference_id": ref_id,
                        "pipeline_run_id": db_ref.pipeline_run_id,
                        "target_type": db_ref.target_type,
                        "target_id": db_ref.target_id,
                    },
                )

            self.session.commit()
            return ref_id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("PipelineReferenceInsertFailed", {"error": str(e)})
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

    def get_by_pipeline_run(self, pipeline_run_id: int) -> List[PipelineReferenceORM]:
        """
        Fetch all references associated with a pipeline run.
        """
        return (
            self.session.query(PipelineReferenceORM)
            .filter(PipelineReferenceORM.pipeline_run_id == pipeline_run_id)
            .order_by(PipelineReferenceORM.created_at.asc())
            .all()
        )

    def get_by_target(self, target_type: str, target_id: str) -> List[PipelineReferenceORM]:
        """
        Fetch all references to a specific scorable target.
        """
        return (
            self.session.query(PipelineReferenceORM)
            .filter(PipelineReferenceORM.target_type == target_type)
            .filter(PipelineReferenceORM.target_id == target_id)
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
            query = query.filter(PipelineReferenceORM.pipeline_run_id == filters["pipeline_run_id"])
        if "target_type" in filters:
            query = query.filter(PipelineReferenceORM.target_type == filters["target_type"])
        if "target_id" in filters:
            query = query.filter(PipelineReferenceORM.target_id == filters["target_id"])
        if "relation_type" in filters:
            query = query.filter(PipelineReferenceORM.relation_type == filters["relation_type"])
        if "since" in filters:
            query = query.filter(PipelineReferenceORM.created_at >= filters["since"])

        return query.order_by(PipelineReferenceORM.created_at.desc()).all()


    def get_documents_by_run_id(self, pipeline_run_id: int, memory, limit: int = 100) -> dict:
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
                scorable = ScorableFactory.from_id(memory, ref.target_type, ref.target_id)
                results[(ref.target_type, ref.target_id)] = scorable
            except Exception as e:
                if self.logger:
                    self.logger.log(
                        "PipelineReferenceResolveFailed",
                        {
                            "pipeline_run_id": pipeline_run_id,
                            "target_type": ref.target_type,
                            "target_id": ref.target_id,
                            "error": str(e),
                        },
                    )

        return results
