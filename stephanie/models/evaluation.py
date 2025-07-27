# stephanie/models/evaluation.py
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (JSON, Column, DateTime, Enum, ForeignKey, Integer,
                        String)
from sqlalchemy.orm import Mapped, relationship

from stephanie.models.base import Base
from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.models.document import DocumentORM
from stephanie.models.evaluation_attribute import EvaluationAttributeORM
from stephanie.models.goal import GoalORM
from stephanie.models.hypothesis import HypothesisORM
from stephanie.models.pipeline_run import PipelineRunORM
from stephanie.models.score import ScoreORM
from stephanie.models.symbolic_rule import SymbolicRuleORM
from stephanie.scoring.scorable_factory import TargetType


class EvaluationORM(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = Column(Integer, primary_key=True)
    
    # Goal reference
    goal_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("goals.id"))
    
    # Polymorphic target reference
    target_type: Mapped[String] = Column(String, nullable=False)
    target_id: Mapped[int] = Column(Integer, nullable=False)
    
    embedding_type: Mapped[Optional[str]] = Column(String, nullable=True)

    # Optional references to other entities
    hypothesis_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("hypotheses.id"))
    document_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("documents.id"))
    symbolic_rule_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("symbolic_rules.id"))
    pipeline_run_id: Mapped[Optional[int]] = Column(Integer, ForeignKey("pipeline_runs.id"))
    belief_cartridge_id = Column(String, ForeignKey("belief_cartridges.id", ondelete="SET NULL"), nullable=True)

    # Metadata
    agent_name: Mapped[str] = Column(String, nullable=False)
    source: Mapped[str] = Column(String)
    model_name: Mapped[str] = Column(String, nullable=False)
    evaluator_name: Mapped[str] = Column(String, nullable=False)
    strategy: Mapped[Optional[str]] = Column(String)
    reasoning_strategy: Mapped[Optional[str]] = Column(String)
    
    # Scores and data
    scores: Mapped[dict] = Column(JSON, default={})
    extra_data: Mapped[Optional[dict]] = Column(JSON)
    
    # Timestamp
    created_at: Mapped[datetime] = Column(
        DateTime, 
        default=lambda: datetime.now(timezone.utc)
    )
    
    # Relationships
    goal: Mapped[Optional[GoalORM]] = relationship("GoalORM", back_populates="evaluations")
    hypothesis: Mapped[Optional[HypothesisORM]] = relationship("HypothesisORM", back_populates="evaluations")
    symbolic_rule: Mapped[Optional[SymbolicRuleORM]] = relationship("SymbolicRuleORM", back_populates="evaluations")
    pipeline_run: Mapped[Optional[PipelineRunORM]] = relationship("PipelineRunORM", back_populates="evaluations")
    document: Mapped[Optional[DocumentORM]] = relationship("DocumentORM", back_populates="evaluations")
    dimension_scores: Mapped[List[ScoreORM]] = relationship(
        "ScoreORM", 
        back_populates="evaluation",
        cascade="all, delete-orphan"
    )
    belief_cartridge: Mapped[Optional["BeliefCartridgeORM"]] = relationship(
        "BeliefCartridgeORM",
        back_populates="evaluations"
    )

    attributes: Mapped[List[EvaluationAttributeORM]] = relationship(
        "EvaluationAttributeORM",
        back_populates="evaluation",
        cascade="all, delete-orphan"
    )

    def to_dict(self, include_relationships: bool = False) -> dict:
        data = {
            "id": self.id,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "document_id": self.document_id,
            "symbolic_rule_id": self.symbolic_rule_id,
            "pipeline_run_id": self.pipeline_run_id,
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "evaluator_name": self.evaluator_name,
            "strategy": self.strategy,
            "reasoning_strategy": self.reasoning_strategy,
            "scores": self.scores,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "embedding_type": self.embedding_type,
            "belief_cartridge_id": self.belief_cartridge_id,
            "attributes": [attr.to_dict() for attr in self.attributes] if self.attributes else []    
        }

        if include_relationships:
            if self.goal:
                data["goal"] = self.goal.to_dict() if hasattr(self.goal, "to_dict") else str(self.goal)
            if self.hypothesis:
                data["hypothesis"] = self.hypothesis.to_dict() if hasattr(self.hypothesis, "to_dict") else str(self.hypothesis)
            if self.document:
                data["document"] = self.document.to_dict() if hasattr(self.document, "to_dict") else str(self.document)
            if self.symbolic_rule:
                data["symbolic_rule"] = self.symbolic_rule.to_dict() if hasattr(self.symbolic_rule, "to_dict") else str(self.symbolic_rule)
            if self.pipeline_run:
                data["pipeline_run"] = self.pipeline_run.to_dict() if hasattr(self.pipeline_run, "to_dict") else str(self.pipeline_run)

        return data