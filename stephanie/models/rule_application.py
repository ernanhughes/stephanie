# models/rule_application.py

from datetime import datetime, timezone

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from .base import Base


class RuleApplicationORM(Base):
    __tablename__ = "rule_applications"

    id = Column(Integer, primary_key=True)
    rule_id = Column(Integer, ForeignKey("symbolic_rules.id", ondelete="CASCADE"))
    goal_id = Column(Integer, ForeignKey("goals.id", ondelete="CASCADE"))
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"))
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"), nullable=True)
    context_hash = Column(String, index=True)

    applied_at = Column(DateTime, default=datetime.now(timezone.utc))
    agent_name = Column(String, nullable=True)
    change_type = Column(String, nullable=True)  # e.g., pipeline_override, hint, etc.
    details = Column(JSON, nullable=True)        # Can store {old:..., new:...}
    stage_details = Column(JSON, nullable=True)

    post_score = Column(Float, nullable=True)
    pre_score = Column(Float, nullable=True)
    delta_score = Column(Float, nullable=True)
    evaluator_name = Column(String, nullable=True)
    rationale = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships (optional, if you want to access related objects)
    goal = relationship("GoalORM", back_populates="rule_applications")
    rule = relationship("SymbolicRuleORM", back_populates="applications")
    pipeline_run = relationship("PipelineRunORM", back_populates="rule_applications")
    hypothesis = relationship("HypothesisORM", back_populates="rule_applications")


    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "goal_id": self.goal_id,
            "pipeline_run_id": self.pipeline_run_id,
            "hypothesis_id": self.hypothesis_id,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "agent_name": self.agent_name,
            "change_type": self.change_type,
            "details": self.details,
            "post_score": self.post_score,
            "pre_score": self.pre_score,
            "delta_score": self.delta_score,
            "evaluator_name": self.evaluator_name,
            "rationale": self.rationale,
            "notes": self.notes
        }
