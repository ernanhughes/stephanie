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
    reports = relationship("ReportORM", back_populates="pipeline_run", cascade="all, delete-orphan")
    theorems = relationship("TheoremORM", back_populates="pipeline_run", cascade="all, delete-orphan")
    cartridges = relationship("CartridgeORM", back_populates="pipeline_run", cascade="all, delete-orphan")
    plan_traces = relationship("PlanTraceORM", back_populates="pipeline_run")
    execution_steps = relationship("ExecutionStepORM", back_populates="pipeline_run")

    def __repr__(self):
        return (
            f"<PipelineRunORM(id={self.id}, "
            f"name='{self.name}', "
            f"description='{(self.description[:50] + '...') if self.description and len(self.description) > 50 else self.description}')>"
        )

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "goal_id": self.goal_id,
            "pipeline": self.pipeline,
            "name": self.name,
            "tag": self.tag,
            "description": self.description,
            "strategy": self.strategy,
            "model_name": self.model_name,
            "embedding_type": self.embedding_type,
            "embedding_dimensions": self.embedding_dimensions,
            "run_config": self.run_config,
            "lookahead_context": self.lookahead_context,
            "symbolic_suggestion": self.symbolic_suggestion,
            "extra_data": self.extra_data,
            "created_at": self.created_at,
        }
