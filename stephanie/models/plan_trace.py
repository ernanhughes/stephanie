# stephanie/models/plan_trace.py (or a suitable path in your models directory)
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (ARRAY, JSON, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stephanie.models.base import Base
# Assuming EmbeddingORM exists (you mentioned a separate table)
from stephanie.models.evaluation import \
    EvaluationORM  # Assuming EvaluationORM exists
# Assuming GoalORM exists
from stephanie.models.goal import GoalORM
from stephanie.models.pipeline_run import PipelineRunORM
from stephanie.models.plan_trace_revision import PlanTraceRevisionORM


# If EmbeddingORM is not directly importable or you just need the ID:
# You can define a foreign key without the full ORM relationship if not needed for navigation here.


class PlanTraceORM(Base):
    """
    ORM to store metadata and key data for a PlanTrace object.
    """

    __tablename__ = "plan_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    pipeline_run_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    pipeline_run: Mapped[Optional["PipelineRunORM"]] = relationship(
        "PipelineRunORM", back_populates="plan_traces"
    )

    trace_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)

    goal_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("goals.id"), nullable=True
    )
    goal: Mapped[Optional["GoalORM"]] = relationship(
        "GoalORM", back_populates="plan_traces"
    )

    revisions: Mapped[List["PlanTraceRevisionORM"]] = relationship(
        "PlanTraceRevisionORM",
        back_populates="plan_trace",
        cascade="all, delete-orphan",
        order_by="PlanTraceRevisionORM.created_at"
    )

    goal_embedding_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    plan_signature: Mapped[str] = mapped_column(String, nullable=False)

    final_output_text: Mapped[str] = mapped_column(Text, nullable=False)
    final_output_embedding_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # --- Epistemic Quality ---
    target_epistemic_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_epistemic_quality_source: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # --- Extended Cognitive Metadata ---
    retrieved_cases: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    strategy_used: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reward_signal: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    skills_used: Mapped[List[str]] = mapped_column(JSON, default=list)
    repair_links: Mapped[List[str]] = mapped_column(JSON, default=list)

    domains: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)  # <-- NEW

    # --- Metadata ---
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})
    extra_data: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # --- Relationships ---
    execution_steps: Mapped[List["ExecutionStepORM"]] = relationship(
        "ExecutionStepORM",
        back_populates="plan_trace",
        cascade="all, delete-orphan",
        order_by="ExecutionStepORM.step_order",
    )

    def to_dict(self, include_steps: bool = False, include_goal: bool = False) -> dict:
        data = {
            "id": self.id,
            "trace_id": self.trace_id,
            "goal_id": self.goal_id,
            "goal_embedding_id": self.goal_embedding_id,
            "plan_signature": self.plan_signature,
            "final_output_text": self.final_output_text,
            "final_output_embedding_id": self.final_output_embedding_id,
            "target_epistemic_quality": self.target_epistemic_quality,
            "target_epistemic_quality_source": self.target_epistemic_quality_source,
            "retrieved_cases": self.retrieved_cases,
            "strategy_used": self.strategy_used,
            "reward_signal": self.reward_signal,
            "skills_used": self.skills_used,
            "repair_links": self.repair_links,
            "meta": self.meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_steps:
            data["execution_steps"] = [step.to_dict() for step in self.execution_steps]
        if include_goal and self.goal:
            data["goal"] = self.goal.to_dict()
        return data

class ExecutionStepORM(Base):
    """
    ORM to store metadata for a single step within a PlanTrace.
    Detailed outputs and scores are linked via EvaluationORM.
    """

    __tablename__ = "execution_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Link to the parent PlanTrace
    plan_trace_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("plan_traces.id"), nullable=False
    )
    plan_trace: Mapped["PlanTraceORM"] = relationship(
        "PlanTraceORM", back_populates="execution_steps"
    )

    # Optional: direct link to pipeline run
    pipeline_run_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
        nullable=True, index=True
    )
    pipeline_run: Mapped[Optional["PipelineRunORM"]] = relationship("PipelineRunORM")

    # Order of the step within the trace
    step_order: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Unique identifier for the step (matches ExecutionStep.step_id)
    step_id: Mapped[str] = mapped_column(String, nullable=False)

    step_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)


    # Description of the step
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Output text of the step
    output_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Output embedding reference (optional)
    output_embedding_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # --- NEW: Agent role ---
    agent_role: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, index=True
    )
    # e.g. "retrieve", "reuse", "revise", "retain"

    # --- Relationships to Scoring ---
    evaluation_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("evaluations.id"), nullable=True, unique=True
    )
    evaluation: Mapped[Optional["EvaluationORM"]] = relationship(
        "EvaluationORM", foreign_keys=[evaluation_id]
    )

    # --- Metadata ---
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    def to_dict(self, include_evaluation: bool = False) -> dict:
        data = {
            "id": self.id,
            "plan_trace_id": self.plan_trace_id,
            "pipeline_run_id": self.pipeline_run_id,
            "step_order": self.step_order,
            "step_id": self.step_id,
            "description": self.description,
            "output_text": self.output_text,
            "output_embedding_id": self.output_embedding_id,
            "evaluation_id": self.evaluation_id,
            "agent_role": self.agent_role,  # ✅ include in export
            "meta": self.meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_evaluation and self.evaluation:
            data["evaluation"] = self.evaluation.to_dict(include_relationships=False)
        return data
