from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String)

from stephanie.models.base import Base


class MARSConflictORM(Base):
    __tablename__ = "mars_conflicts"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Links to run and trace
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=True)
    plan_trace_id = Column(String, ForeignKey("plan_traces.trace_id", ondelete="CASCADE"), nullable=True)

    # Conflict metadata
    dimension = Column(String, nullable=False)
    primary_conflict = Column(JSON, nullable=False)   # e.g. ["mrq", "ebt"]
    delta = Column(Float, nullable=False)             # magnitude of disagreement
    agreement_score = Column(Float, nullable=True)
    preferred_model = Column(String, nullable=True)
    explanation = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "pipeline_run_id": self.pipeline_run_id,
            "plan_trace_id": self.plan_trace_id,
            "dimension": self.dimension,
            "primary_conflict": self.primary_conflict,
            "delta": self.delta,
            "agreement_score": self.agreement_score,
            "preferred_model": self.preferred_model,
            "explanation": self.explanation,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return (
            f"<MARSConflictORM(id={self.id}, dimension={self.dimension}, "
            f"primary_conflict={self.primary_conflict}, delta={self.delta})>"
        )
