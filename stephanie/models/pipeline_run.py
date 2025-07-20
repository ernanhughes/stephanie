# stephanie/models/pipeline_run.py
from datetime import datetime, timezone

from sqlalchemy import (JSON, Column, DateTime, ForeignKey, Integer, String,
                        Text)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class PipelineRunORM(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, nullable=False)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False)
    pipeline = Column(JSON)  # Stored as JSONB or TEXT[]
    name = Column(String)
    tag = Column(String)
    description = Column(String)
    strategy = Column(String)    
    model_name = Column(String)
    embedding_type = Column(Text, nullable=True)
    embedding_dimensions = Column(Integer, nullable=True)
    run_config = Column(JSON)
    lookahead_context = Column(JSON)
    symbolic_suggestion = Column(JSON)
    extra_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    goal = relationship("GoalORM", back_populates="pipeline_runs")
    prompts = relationship(
        "PromptORM", back_populates="pipeline_run", cascade="all, delete-orphan"
    )
    hypotheses = relationship("HypothesisORM", back_populates="pipeline_run")
    symbolic_rules = relationship("SymbolicRuleORM", back_populates="pipeline_run")
    prompt_programs = relationship("PromptProgramORM", back_populates="pipeline_run")

    rule_applications = relationship(
        "RuleApplicationORM",
        back_populates="pipeline_run",
        cascade="all, delete-orphan",
    )
    evaluations = relationship("EvaluationORM", back_populates="pipeline_run")
    stages = relationship("PipelineStageORM", back_populates="pipeline_run")
