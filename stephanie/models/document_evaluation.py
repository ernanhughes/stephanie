# models/document_evaluation.py

from datetime import datetime, timezone

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class DocumentEvaluationORM(Base):
    __tablename__ = "document_evaluations"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=True)
    agent_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    evaluator_name = Column(String, nullable=False)
    strategy = Column(String)
    reasoning_strategy = Column(String)

    scores = Column(JSON, default={})
    extra_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    dimension_scores = relationship(
        "DocumentScoreORM", back_populates="evaluation", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "document_id": self.document_id,
            "pipeline_run_id": self.pipeline_run_id,
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "evaluator_name": self.evaluator_name,
            "strategy": self.strategy,
            "reasoning_strategy": self.reasoning_strategy,
            "scores": self.scores,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
