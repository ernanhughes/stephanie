# stephanie/models/plan_trace.py (or a suitable path in your models directory)

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from stephanie.models.base import Base

# Assuming GoalORM exists
from stephanie.models.goal import GoalORM

# Assuming EmbeddingORM exists (you mentioned a separate table)
from stephanie.models.evaluation import (
    EvaluationORM,
)  # Assuming EvaluationORM exists
# If EmbeddingORM is not directly importable or you just need the ID:
# You can define a foreign key without the full ORM relationship if not needed for navigation here.


class PlanTraceORM(Base):
    """
    ORM to store metadata and key data for a PlanTrace object.
    The full serialized PlanTrace (including steps) can be stored separately
    (e.g., in a file or a large text/blob column) and linked via path or ID.
    This ORM focuses on queryable metadata and relationships.
    """

    __tablename__ = "plan_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Unique identifier for the trace (matches PlanTrace.trace_id)
    trace_id: Mapped[str] = mapped_column(
        String, unique=True, index=True, nullable=False
    )

    # Link to the original goal
    goal_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("goals.id"), nullable=True
    )
    goal: Mapped[Optional[GoalORM]] = relationship(
        "GoalORM", back_populates="plan_traces"
    )  # Add plan_traces to GoalORM

    # Goal embedding reference (assuming EmbeddingORM exists)
    # goal_embedding_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("embeddings.id"), nullable=True)
    # If you just need the ID without ORM relationship for now:
    goal_embedding_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    # Signature of the plan that generated this trace
    plan_signature: Mapped[str] = mapped_column(String, nullable=False)

    # Final output text (cached for easy access)
    final_output_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Final output embedding reference (assuming EmbeddingORM exists)
    # final_output_embedding_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("embeddings.id"), nullable=True)
    # If you just need the ID without ORM relationship for now:
    final_output_embedding_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    # --- Target for Epistemic Plan HRM Training ---
    # The score the HRM model should predict for this trace's epistemic quality.
    target_epistemic_quality: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    # Source of the target quality score (e.g., "llm_judgment", "proxy_metric_avg_sicql_q")
    target_epistemic_quality_source: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )

    # --- Metadata ---
    # JSON blob for flexible metadata (matches PlanTrace.meta)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})
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
    # Link to Execution Steps
    execution_steps: Mapped[List["ExecutionStepORM"]] = relationship(
        "ExecutionStepORM",
        back_populates="plan_trace",
        cascade="all, delete-orphan",  # If a trace is deleted, delete its steps
        order_by="ExecutionStepORM.step_order",  # Order steps by their sequence
    )

    # Optional: Link to the EvaluationORM representing the *scoring* of the final output
    # This connects the trace's outcome to the standard scoring system.
    # final_evaluation_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("evaluations.id"), nullable=True)
    # final_evaluation: Mapped[Optional["EvaluationORM"]] = relationship("EvaluationORM", foreign_keys=[final_evaluation_id])

    def to_dict(
        self, include_steps: bool = False, include_goal: bool = False
    ) -> dict:
        """Convert ORM object to dictionary, optionally including related data."""
        data = {
            "id": self.id,
            "trace_id": self.trace_id,
            "goal_id": self.goal_id,
            "goal_text": self.goal_text,
            "goal_embedding_id": self.goal_embedding_id,
            "plan_signature": self.plan_signature,
            "final_output_text": self.final_output_text,
            "final_output_embedding_id": self.final_output_embedding_id,
            "target_epistemic_quality": self.target_epistemic_quality,
            "target_epistemic_quality_source": self.target_epistemic_quality_source,
            "meta": self.meta,
            "created_at": self.created_at.isoformat()
            if self.created_at
            else None,
            "updated_at": self.updated_at.isoformat()
            if self.updated_at
            else None,
        }
        if include_steps:
            data["execution_steps"] = [
                step.to_dict() for step in self.execution_steps
            ]
        if include_goal and self.goal:
            data["goal"] = self.goal.to_dict()
        return data


# Add this relationship to your GoalORM class definition:
# class GoalORM(Base):
#     # ... existing fields ...
#     plan_traces: Mapped[List[PlanTraceORM]] = relationship("PlanTraceORM", back_populates="goal")


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

    # Order of the step within the trace
    step_order: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True
    )

    # Unique identifier for the step (matches ExecutionStep.step_id)
    step_id: Mapped[str] = mapped_column(String, nullable=False)

    # Description of the step
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Output text of the step
    output_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Output embedding reference (assuming EmbeddingORM exists)
    # output_embedding_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("embeddings.id"), nullable=True)
    # If you just need the ID without ORM relationship for now:
    output_embedding_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    # --- Relationships to Scoring ---
    # Link to the EvaluationORM representing the *scoring* of this step's output.
    # This is the standard way scores are stored in Stephanie.
    evaluation_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("evaluations.id"), nullable=True, unique=True
    )
    evaluation: Mapped[Optional["EvaluationORM"]] = relationship(
        "EvaluationORM",
        foreign_keys=[evaluation_id],
        # back_populates="plan_trace_step" # Add this to EvaluationORM if needed for reverse nav
    )

    # Optional: Store simplified scores directly if needed for quick access
    # (Though EvaluationAttributeORM via evaluation is preferred)
    # cached_sicql_q: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # cached_ebt_energy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # --- Metadata ---
    # JSON blob for flexible step-specific metadata (matches ExecutionStep.meta)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    def to_dict(self, include_evaluation: bool = False) -> dict:
        """Convert ORM object to dictionary."""
        data = {
            "id": self.id,
            "plan_trace_id": self.plan_trace_id,
            "step_order": self.step_order,
            "step_id": self.step_id,
            "description": self.description,
            "output_text": self.output_text,
            "output_embedding_id": self.output_embedding_id,
            "evaluation_id": self.evaluation_id,
            "meta": self.meta,
            "created_at": self.created_at.isoformat()
            if self.created_at
            else None,
        }
        if include_evaluation and self.evaluation:
            # Be careful with recursion; EvaluationORM.to_dict might include steps
            data["evaluation"] = self.evaluation.to_dict(
                include_relationships=False
            )
        return data
