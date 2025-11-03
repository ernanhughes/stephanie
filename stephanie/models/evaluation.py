# stephanie/models/evaluation.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String)
from sqlalchemy.orm import Mapped, relationship

from stephanie.models.base import Base
from stephanie.models.evaluation_attribute import EvaluationAttributeORM
from stephanie.models.goal import GoalORM
from stephanie.models.pipeline_run import PipelineRunORM
from stephanie.models.score import ScoreORM
from stephanie.models.symbolic_rule import SymbolicRuleORM


class EvaluationORM(Base):
    """
    Unified evaluation record.

    Each evaluation attaches scores and metadata to a `scorable`
    (document, hypothesis, plan trace, cartridge, etc.) without
    hardcoding specific foreign keys.
    """

    __tablename__ = "evaluations"

    id: Mapped[int] = Column(Integer, primary_key=True)

    # Goal reference (optional)
    goal_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("goals.id"))
    plan_trace_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("plan_traces.id", ondelete="CASCADE"), index=True, nullable=True)


    scorable_type: Mapped[str] = Column(String, nullable=False, index=True)
    scorable_id: Mapped[str] = Column(String, nullable=False, index=True)

    # Optional: query reference (what this scorable was evaluated against)
    query_type: Mapped[Optional[str]] = Column(String, nullable=True, index=True)
    query_id: Mapped[Optional[str]] = Column(String, nullable=True, index=True)

    embedding_type: Mapped[Optional[str]] = Column(String, nullable=True)

    # Optional references to other entities
    symbolic_rule_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("symbolic_rules.id"))
    pipeline_run_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("pipeline_runs.id"))
    
    # Metadata
    agent_name: Mapped[str] = Column(String, nullable=False)
    source: Mapped[Optional[str]] = Column(String)
    model_name: Mapped[str] = Column(String, nullable=False)
    evaluator_name: Mapped[str] = Column(String, nullable=False)
    strategy: Mapped[Optional[str]] = Column(String)
    reasoning_strategy: Mapped[Optional[str]] = Column(String)

    # Scores and extra details
    scores: Mapped[Dict[str, Any]] = Column(JSON, default={})
    extra_data: Mapped[Optional[Dict[str, Any]]] = Column(JSON)

    # Timestamp
    created_at: Mapped[datetime] = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    goal: Mapped[Optional[GoalORM]] = relationship("GoalORM", back_populates="evaluations")
    symbolic_rule: Mapped[Optional[SymbolicRuleORM]] = relationship("SymbolicRuleORM", back_populates="evaluations")
    pipeline_run: Mapped[Optional[PipelineRunORM]] = relationship("PipelineRunORM", back_populates="evaluations")

    dimension_scores: Mapped[List[ScoreORM]] = relationship(
        "ScoreORM",
        back_populates="evaluation",
        cascade="all, delete-orphan",
    )

    attributes: Mapped[List[EvaluationAttributeORM]] = relationship(
        "EvaluationAttributeORM",
        back_populates="evaluation",
        cascade="all, delete-orphan",
    )

    def to_dict(self, include_relationships: bool = False) -> dict:
        data = {
            "id": self.id,
            "goal_id": self.goal_id,
            "scorable_type": self.scorable_type,
            "scorable_id": self.scorable_id,
            "query_type": self.query_type,
            "query_id": self.query_id,
            "agent_name": self.agent_name,
            "source": self.source,
            "model_name": self.model_name,
            "evaluator_name": self.evaluator_name,
            "strategy": self.strategy,
            "reasoning_strategy": self.reasoning_strategy,
            "embedding_type": self.embedding_type,
            "scores": self.scores,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "attributes": [attr.to_dict() for attr in self.attributes] if self.attributes else [],
        }

        if include_relationships:
            if self.goal:
                data["goal"] = self.goal.to_dict()
            if self.pipeline_run:
                data["pipeline_run"] = self.pipeline_run.to_dict()

        return data
