# stephanie/orm/mars_result.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String)

from stephanie.orm.base import Base


class MARSResultORM(Base):
    __tablename__ = "mars_results"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to pipeline run and/or plan trace
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=True)
    plan_trace_id = Column(String, ForeignKey("plan_traces.trace_id", ondelete="CASCADE"), nullable=True)

    # Dimension scored (e.g., relevance, clarity, epistemic_quality)
    dimension = Column(String, nullable=False)
    source = Column(String, nullable=True)

    average_score = Column(Float, nullable=True)
    agreement_score = Column(Float, nullable=False)
    std_dev = Column(Float, nullable=False)
    preferred_model = Column(String, nullable=True)
    primary_conflict = Column(JSON, nullable=True)
    delta = Column(Float, nullable=True)

    high_disagreement = Column(Integer, nullable=False, default=0)  # 0/1 flag
    explanation = Column(String, nullable=True)

    # Extended metrics
    scorer_metrics = Column(JSON, nullable=True)
    metric_correlations = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "pipeline_run_id": self.pipeline_run_id,
            "plan_trace_id": self.plan_trace_id,
            "dimension": self.dimension,
            "source": self.source,
            "agreement_score": self.agreement_score,
            "std_dev": self.std_dev,
            "preferred_model": self.preferred_model,
            "primary_conflict": self.primary_conflict,
            "delta": self.delta,
            "high_disagreement": bool(self.high_disagreement),
            "explanation": self.explanation,
            "scorer_metrics": self.scorer_metrics,
            "metric_correlations": self.metric_correlations,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<MARSResultORM(id={self.id}, pipeline_run_id={self.pipeline_run_id}, dimension={self.dimension}, agreement_score={self.agreement_score})>"
