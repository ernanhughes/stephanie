# stephanie/orm/measurement.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String, Text)
from sqlalchemy.orm import relationship

from stephanie.orm.base import Base


class MeasurementORM(Base):
    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True)

    # Target entity being measured
    entity_type = Column(Text, nullable=False)  # e.g., "Cartridge", "Theorem", "Prompt"
    entity_id = Column(Integer, nullable=False)  # ID of the measured object

    # Type of measurement
    metric_name = Column(
        Text, nullable=False
    )  # e.g., "domain_density", "semantic_coverage"

    # Value(s) of the measurement
    value = Column(JSON)  # Could be float, list, dict, etc.

    # Contextual metadata
    context = Column(JSON)  # Optional: goal_id, session_id, etc.

    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return (
            f"<Measurement {self.metric_name} on {self.entity_type}[{self.entity_id}]>"
        )

# stephanie/orm/measurement_quality.py
class MeasurementQualityORM(Base):
    """
    Quality assessment for a measurement.
    Decides whether the measurement is usable for training / updates.
    """
    __tablename__ = "measurement_quality"

    id = Column(String, primary_key=True)  # universal id

    measurement_id = Column(String, ForeignKey("measurements.id"), index=True, nullable=False)
    measurement = relationship("MeasurementORM", back_populates="quality")

    # 0–1 or 0–100 quality score (LLM/HRM/heuristic)
    quality_score = Column(Float, nullable=False)

    # Simple gate: should we use this for training?
    is_usable_for_training = Column(Boolean, default=False, index=True)

    # Who/what did the quality check?
    assessor_type = Column(String, index=True)  # "rule", "llm", "hrm", "ensemble", ...
    assessor_model_name = Column(String, nullable=True)
    assessor_model_version = Column(String, nullable=True)

    # Why did we decide this?
    rationale = Column(String)       # short text
    details = Column(JSON, default=dict)   # per-rule breakdown, feature flags, etc.

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
