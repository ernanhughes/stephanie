# models/prompt.py
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from co_ai.models.base import Base


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
