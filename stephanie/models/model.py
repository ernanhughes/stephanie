# stephanie/models/model.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class ModelORM(Base):
    """
    Conceptual model registry entry.

    One row per (model_type, target_type, dimension, score_mode) combo.
    Tracks which version is active and holds high-level metadata.
    """
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identity for this logical model
    model_type = Column(Text, nullable=False)      # "sicql", "hrm", "tiny", "mrq", "llm", ...
    target_type = Column(Text, nullable=False)     # "document", "plan_trace", "code_file", ...
    dimension = Column(Text, nullable=False)       # "helpfulness", "correctness", ...
    score_mode = Column(Text, nullable=True)       # "reward", "rank", "risk", etc.

    # Active version string, matching ModelVersionORM.version
    active_version = Column(Text, nullable=True, index=True)

    # High-level status for lifecycle control
    status = Column(
        String,
        default="unknown",
        nullable=False,
        index=True,
    )  # "unknown" | "ready" | "training" | "stale" | "degraded" | "disabled" | "error"

    description = Column(Text, nullable=True)
    meta = Column(JSON, default={})

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, nullable=False)

    # Relationships
    health = relationship("ModelHealthORM", back_populates="model", uselist=False)

    __table_args__ = (
        UniqueConstraint(
            "model_type",
            "target_type",
            "dimension",
            "score_mode",
            name="uq_models_type_target_dim_score_mode",
        ),
        Index(
            "ix_models_model_type_target_dim",
            "model_type",
            "target_type",
            "dimension",
        ),
    )

    def key(self) -> str:
        """Stable string identifier you can reuse in logs/meta."""
        parts = [self.model_type, self.target_type, self.dimension]
        if self.score_mode:
            parts.append(self.score_mode)
        return "/".join(parts)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_type": self.model_type,
            "target_type": self.target_type,
            "dimension": self.dimension,
            "score_mode": self.score_mode,
            "active_version": self.active_version,
            "status": self.status,
            "description": self.description,
            "meta": self.meta or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ModelHealthORM(Base):
    """
    Current health/drift summary for a logical model.

    This is updated by your drift analysis / training supervisor and read by
    the locator / policy layer to decide 'can I trust this model?'.
    """
    __tablename__ = "model_health"

    id = Column(Integer, primary_key=True, autoincrement=True)

    model_id = Column(
        Integer,
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    model = relationship("ModelORM", back_populates="health")

    # High-level status for health, separate from lifecycle status if you want
    # e.g., "healthy" but "training" in progress, etc.
    health_status = Column(
        String,
        default="unknown",
        nullable=False,
        index=True,
    )  # "unknown" | "healthy" | "drifting" | "degraded" | "offline" | "experimental"

    # Aggregate drift / quality metrics (all optional, set by your drift jobs)
    drift_score = Column(Float, nullable=True)          # e.g. EWMA of divergence_score
    mean_delta = Column(Float, nullable=True)
    mean_abs_delta = Column(Float, nullable=True)
    sign_flip_ratio = Column(Float, nullable=True)      # fraction of sign flips
    outlier_ratio = Column(Float, nullable=True)        # fraction of 'is_outlier' divergences

    # Data freshness / coverage
    num_measurements = Column(Integer, nullable=True)
    num_training_examples = Column(Integer, nullable=True)
    data_freshness_days = Column(Float, nullable=True)  # since last training data batch

    # Lifecycle timestamps
    last_retrain_at = Column(DateTime, nullable=True)
    last_evaluated_at = Column(DateTime, default=datetime.now, nullable=False)

    # Arbitrary structured metrics (per-dimension, per-window, etc.)
    metrics = Column(JSON, default={})
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "health_status": self.health_status,
            "drift_score": self.drift_score,
            "mean_delta": self.mean_delta,
            "mean_abs_delta": self.mean_abs_delta,
            "sign_flip_ratio": self.sign_flip_ratio,
            "outlier_ratio": self.outlier_ratio,
            "num_measurements": self.num_measurements,
            "num_training_examples": self.num_training_examples,
            "data_freshness_days": self.data_freshness_days,
            "last_retrain_at": self.last_retrain_at.isoformat() if self.last_retrain_at else None,
            "last_evaluated_at": self.last_evaluated_at.isoformat() if self.last_evaluated_at else None,
            "metrics": self.metrics or {},
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
