import hashlib
import json
from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from .base import Base


class SymbolicRuleORM(Base):
    __tablename__ = "symbolic_rules"

    id = Column(Integer, primary_key=True)

    # General metadata
    source = Column(String)  # e.g., 'manual', 'lookahead', 'pipeline_stage'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # New design: generalized rules
    target = Column(String, nullable=False)  # e.g., 'goal', 'agent', 'prompt', 'pipeline', 'hypothesis'
    attributes = Column(JSON)  # What to apply, e.g., {"difficulty": "hard"}
    filter = Column(JSON)  # How to match, e.g., {"goal_type": "research", "strategy": "reasoning"}
    context_hash = Column(String, index=True)  # Hash of (filters + attributes)


    goal_type = Column(String)
    goal_category = Column(String)
    difficulty = Column(String)
    focus_area = Column(String)

    # Optional linkage (for legacy/rule provenance/debugging)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=True)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=True)
    agent_name = Column(String, nullable=True)

    # Optional scoring/description
    score = Column(Float)
    rule_text = Column(Text)

    # Relationships
    goal = relationship("GoalORM", back_populates="symbolic_rules", lazy="joined")
    pipeline_run = relationship("PipelineRunORM", back_populates="symbolic_rules", lazy="joined")
    prompt = relationship("PromptORM", back_populates="symbolic_rules", lazy="joined")
    scores = relationship("EvaluationORM", back_populates="symbolic_rule")
    applications = relationship(
        "RuleApplicationORM", back_populates="rule", cascade="all, delete-orphan"
    )

    def __str__(self):
        return (
            f"<SymbolicRuleORM id={self.id} "
            f"target={self.target} "
            f"filter={json.dumps(self.filter, sort_keys=True)} "
            f"attributes={json.dumps(self.attributes, sort_keys=True)} "
            f"context_hash={self.context_hash[:8]}... "
            f"source={self.source} "
            f"agent={self.agent_name}>"
        )

    @staticmethod
    def compute_context_hash(filters: dict, attributes: dict) -> str:
        merged = {"filters": filters or {}, "attributes": attributes or {}}
        canonical_str = json.dumps(merged, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self, include_relationships=False):
        return {
            "id": self.id,
            "target": self.target,
            "filter": self.filter,
            "attributes": self.attributes,
            "context_hash": self.context_hash,
            "source": self.source,
            "goal_id": self.goal_id,
            "pipeline_run_id": self.pipeline_run_id,
            "prompt_id": self.prompt_id,
            "agent_name": self.agent_name,
            "score": self.score,
            "rule_text": self.rule_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
