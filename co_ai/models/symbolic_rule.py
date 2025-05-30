# models/symbolic_rule.py
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship

from .base import Base



class SymbolicRuleORM(Base):
    __tablename__ = "symbolic_rules"

    id = Column(Integer, primary_key=True)

    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=True)
    rule_text = Column(String)
    agent_name = Column(String)
    source = Column(String)
    goal_type = Column(String)
    goal_category = Column(String)
    difficulty = Column(String)
    focus_area = Column(String)
    score = Column(Float)

    goal = relationship("GoalORM", back_populates="symbolic_rules")
    pipeline_run = relationship("PipelineRunORM", back_populates="symbolic_rules")
    prompt = relationship("PromptORM", back_populates="symbolic_rules")
    scores = relationship("ScoreORM", back_populates="symbolic_rule")
    rule_applications = relationship("RuleApplicationORM",  back_populates="rule", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<SymbolicRule(agent={self.agent_name}, pipeline={self.pipeline_signature}, score={self.score:.2f})>"
