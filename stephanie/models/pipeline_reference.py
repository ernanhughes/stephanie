# stephanie/models/pipeline_reference.py
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from stephanie.models.base import Base


class PipelineReferenceORM(Base):
    """
    Maps a pipeline run to any referenced scorable object.

    - A pipeline run may reference many objects (documents, prompts, traces, etc.).
    - A scorable object may be referenced by many runs.
    - References are stored by target_type + target_id, not FKs, to stay polymorphic.
    """

    __tablename__ = "pipeline_references"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # The pipeline run that made the reference
    pipeline_run_id = Column(Integer, nullable=False, index=True)

    # Polymorphic target
    scorable_type = Column(String, nullable=False)  # e.g. "document", "plan_trace"
    scorable_id = Column(String, nullable=False)    # foreign entity's id, stored as string

    # Optional: why/how it was referenced
    relation_type = Column(String, nullable=True)
    source = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # --- Helper method ---
    def resolve(self, memory, mode: str = "default"):
        """
        Resolve this reference to a Scorable via ScorableFactory.
        Uses memory managers to load the ORM object.
        """
        from stephanie.scoring.scorable_factory import ScorableFactory

        return ScorableFactory.from_id(
            memory=memory, scorable_type=self.scorable_type, scorable_id=self.scorable_id
        )
