# stephanie/models/model_divergence.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class ModelDivergenceORM(Base):
    """
    Divergence between a new measurement and an existing model's belief.

    This is the primitive for drift detection and 'we need to retrain' logic.
    """
    __tablename__ = "model_divergence"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to the logical model (ModelORM)
    logical_model_id = Column(
        Integer,
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    measurement_id = Column(
        String,
        ForeignKey("measurements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    measurement = relationship("MeasurementORM", back_populates="divergences")

    # Which concrete model artifact are we comparing against?
    model_name = Column(String, index=True)       # e.g. SICQL path key or artifact name
    model_version = Column(String, index=True)

    # What the model thought (at comparison time)
    predicted_value = Column(Float, nullable=False)
    # The actual measured value (duplicated for convenience)
    observed_value = Column(Float, nullable=False)

    # Basic deltas
    delta = Column(Float, nullable=False)         # observed - predicted
    abs_delta = Column(Float, nullable=False)
    # Normalized divergence (e.g., z-score, scaled 0–1, etc.)
    divergence_score = Column(Float, nullable=False)

    # Simple flags
    sign_flip = Column(Boolean, default=False)    # e.g., from positive to negative
    is_outlier = Column(Boolean, default=False)   # > N std devs away from trend

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # optional backref if you want
    # model = relationship("ModelORM", back_populates="divergences")


Index(
    "ix_model_divergence_model_time",
    ModelDivergenceORM.logical_model_id,
    ModelDivergenceORM.created_at,
)

Index(
    "ix_model_divergence_artifact",
    ModelDivergenceORM.model_name,
    ModelDivergenceORM.model_version,
)
