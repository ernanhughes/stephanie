# models/prompt.py
import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String, Text)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class PromptORM(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True)

    # Agent and prompt metadata
    goal_id = Column(Integer, ForeignKey("goals.id"))
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True)
    agent_name = Column(String, nullable=False)
    prompt_key = Column(String, nullable=False)  # e.g., generation_goal_aligned.txt
    prompt_text = Column(Text, nullable=False)
    response_text = Column(Text)  # Optional — if storing model output too
    source = Column(String)  # e.g., manual, dsp_refinement, feedback_injection
    strategy = Column(String)  # e.g., goal_aligned, out_of_the_box
    version = Column(Integer, default=1)
    is_current = Column(Boolean, default=False)
    extra_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc))

    goal = relationship("GoalORM", back_populates="prompts")
    hypotheses = relationship("HypothesisORM", back_populates="prompt")
    symbolic_rules = relationship("SymbolicRuleORM", back_populates="prompt")
    pipeline_run = relationship("PipelineRunORM", back_populates="prompts")
    program = relationship("PromptProgramORM", uselist=False, back_populates="prompt")

    def to_dict(self, include_relationships: bool = False) -> dict:
        data = {
            "id": self.id,
            "agent_name": self.agent_name,
            "prompt_key": self.prompt_key,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "goal_id": self.goal_id,
            "source": self.source,
            "strategy": self.strategy,
            "version": self.version,
            "is_current": self.is_current,
            "extra_data": self.extra_data,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

        if include_relationships:
            data["goal"] = self.goal.to_dict() if self.goal else None
            data["hypotheses"] = [h.to_dict() for h in self.hypotheses] if self.hypotheses else []

        return data


def generate_uuid():
    return str(uuid.uuid4())

class PromptProgramORM(Base):
    __tablename__ = "prompt_programs"

    id = Column(String, primary_key=True, default=generate_uuid)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id", ondelete="SET NULL"), nullable=True)
    goal = Column(Text, nullable=False)
    template = Column(Text, nullable=False)
    inputs = Column(JSON, default={})
    version = Column(Integer, default=1)
    parent_id = Column(String, ForeignKey("prompt_programs.id"), nullable=True)
    strategy = Column(String, default="default")
    prompt_text = Column(Text, nullable=True)
    hypothesis = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)
    mutation_type = Column(String, nullable=True)
    execution_trace = Column(Text, nullable=True)
    extra_data = Column(JSON, default={})

 
    parent = relationship("PromptProgramORM", remote_side=[id], backref="children")
    prompt = relationship("PromptORM", back_populates="program")
    pipeline_run = relationship("PipelineRunORM", back_populates="prompt_programs")

    def to_dict(self):
        return {
            "id": self.id,
            "goal": self.goal,
            "template": self.template,
            "inputs": self.inputs,
            "version": self.version,
            "parent_id": self.parent_id,
            "prompt_id": self.prompt_id,
            "propipeline_run_idmpt_id": self.pipeline_run_id,
            "strategy": self.strategy,
            "prompt_text": self.prompt_text,
            "hypothesis": self.hypothesis,
            "score": self.score,
            "rationale": self.rationale,
            "mutation_type": self.mutation_type,
            "execution_trace": self.execution_trace,
            "extra_data": self.extra_data,
        }
