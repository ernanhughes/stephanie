# models/score.py
from datetime import datetime, timezone

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from co_ai.models.base import Base


class EvaluationORM(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey("goals.id"))
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"))
    symbolic_rule_id = Column(Integer, ForeignKey("symbolic_rules.id"), nullable=True)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=True)
    agent_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    evaluator_name = Column(String, nullable=False)
    strategy = Column(String)
    reasoning_strategy = Column(String)
    
    scores = Column(JSON, default={})
    
    extra_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    goal = relationship("GoalORM", back_populates="scores")
    hypothesis = relationship("HypothesisORM", back_populates="scores")
    symbolic_rule = relationship("SymbolicRuleORM", back_populates="scores")
    pipeline_run = relationship("PipelineRunORM", back_populates="scores")
    dimension_scores = relationship("ScoreORM", back_populates="evaluation", cascade="all, delete-orphan")

    def to_dict(self, include_relationships: bool = False) -> dict:
        data = {
            "id": self.id,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
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
        }

        if include_relationships:
            data["goal"] = (
                self.goal.to_dict()
                if self.goal and hasattr(self.goal, "to_dict")
                else None
            )
            data["hypothesis"] = (
                self.hypothesis.to_dict()
                if self.hypothesis and hasattr(self.hypothesis, "to_dict")
                else None
            )

        return data
