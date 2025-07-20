# stephanie/models/hypothesis.py
# models/hypothesis.py
from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class HypothesisORM(Base):
    __tablename__ = "hypotheses"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False)
    goal_id = Column(Integer, ForeignKey("goals.id"))
    prompt_id = Column(Integer, ForeignKey("prompts.id"))
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"))
    source_hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"))
    strategy = Column(String)
    confidence = Column(Float, default=0.0)
    review = Column(String)
    reflection = Column(String)
    elo_rating = Column(Float, default=750.0)
    embedding = Column(JSON)  # Use pgvector later for better support
    features = Column(JSON)  # For structured metadata
    source = Column(String)
    pipeline_signature = Column(String)
    enabled = Column(Boolean, default=True)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    goal = relationship("GoalORM", back_populates="hypotheses")
    prompt = relationship("PromptORM", back_populates="hypotheses")
    source_hypothesis = relationship("HypothesisORM", remote_side=[id])
    evaluations = relationship("EvaluationORM", back_populates="hypothesis")
    pipeline_run = relationship("PipelineRunORM", back_populates="hypotheses")

    rule_applications = relationship(
        "RuleApplicationORM",
        back_populates="hypothesis",
        cascade="all, delete-orphan",
    )

    def to_dict(self, include_related=False):
        return {
            "id": self.id,
            "text": self.text,
            "goal_id": self.goal_id,
            "prompt_id": self.prompt_id,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "review": self.review,
            "reflection": self.reflection,
            "elo_rating": self.elo_rating,
            "embedding": self.embedding,
            "features": self.features,
            "source_hypothesis_id": self.source_hypothesis_id,
            "source": self.source,
            "pipeline_signature": self.pipeline_signature,
            "pipeline_id": self.pipeline_run_id,
            "enabled": self.enabled,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            # Optional related objects
            "goal": self.goal.to_dict() if include_related and self.goal else None,
            "prompt": self.prompt.to_dict()
            if include_related and self.prompt
            else None,
            "scores": [s.to_dict() for s in self.scores]
            if include_related and self.scores
            else None,
        }

    @staticmethod
    def from_dict(data: dict):
        return HypothesisORM(
            id=data.get("id"),
            text=data.get("text", ""),
            goal_id=data.get("goal_id"),
            prompt_id=data.get("prompt_id"),
            strategy=data.get("strategy"),
            confidence=data.get("confidence", 0.0),
            review=data.get("review"),
            reflection=data.get("reflection"),
            elo_rating=data.get("elo_rating", 750.0),
            embedding=data.get("embedding"),
            features=data.get("features"),
            source_hypothesis_id=data.get("source_hypothesis_id"),
            source=data.get("source"),
            pipeline_signature=data.get("pipeline_signature"),
            pipeline_id=data.get("pipeline_id"),
            enabled=data.get("enabled", True),
            version=data.get("version", 1),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
