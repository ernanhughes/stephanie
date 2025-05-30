# models/score.py
from datetime import datetime, timezone

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from .base import Base


class ScoreORM(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey("goals.id"))
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"))
    symbolic_rule_id = Column(Integer, ForeignKey("symbolic_rules.id"), nullable=True)
    agent_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    evaluator_name = Column(String, nullable=False)
    score_type = Column(String, nullable=False)
    score = Column(Float)
    score_text = Column(Text)
    strategy = Column(String)
    reasoning_strategy = Column(String)
    rationale = Column(Text)
    reflection = Column(Text)
    review = Column(Text)
    meta_review = Column(Text)
    run_id = Column(String)
    extra_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    goal = relationship("GoalORM", back_populates="scores")
    hypothesis = relationship("HypothesisORM", back_populates="scores")
    symbolic_rule = relationship("SymbolicRuleORM", back_populates="scores")

    def to_dict(self, include_relationships: bool = False) -> dict:
        data = {
            "id": self.id,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "symbolic_rule_id": self.symbolic_rule_id,
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "evaluator_name": self.evaluator_name,
            "score_type": self.score_type,
            "score": self.score,
            "score_text": self.score_text,
            "strategy": self.strategy,
            "reasoning_strategy": self.reasoning_strategy,
            "rationale": self.rationale,
            "reflection": self.reflection,
            "review": self.review,
            "meta_review": self.meta_review,
            "run_id": self.run_id,
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
