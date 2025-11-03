# stephanie/models/calibration.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer,
                        LargeBinary, String, UniqueConstraint)

from stephanie.models.base import Base


class CalibrationModelORM(Base):
    __tablename__ = "calibration_models"
    __table_args__ = (UniqueConstraint("domain", name="uq_calibration_models_domain"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(255), nullable=False, index=True)
    kind = Column(String(64), nullable=False)          # e.g., "quantile", "logistic_raw", "sigmoid_raw"
    threshold = Column(Float, nullable=False, default=0.5)
    payload = Column(LargeBinary, nullable=False)      # pickle.dumps(calibrator)
    updated_at = Column(DateTime, default=datetime.now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "domain": self.domain,
            "kind": self.kind,
            "threshold": float(self.threshold),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CalibrationEventORM(Base):
    __tablename__ = "calibration_events"

    id = Column(Integer, primary_key=True)
    domain = Column(String, nullable=False)
    query = Column(String, nullable=False)
    raw_similarity = Column(Float, nullable=False)
    scorable_id = Column(String, nullable=False)
    scorable_type = Column(String, nullable=False)
    entity_type = Column(String)
    is_relevant = Column(Boolean, nullable=False)  # Ground truth
    context = Column(JSON)  # Optional: goal, section, etc.
    timestamp = Column(DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "domain": self.domain,
            "query": self.query,
            "raw_similarity": self.raw_similarity,
            "scorable_id": self.scorable_id,
            "is_relevant": self.is_relevant,
            "entity_type": self.entity_type,
            "timestamp": self.timestamp.isoformat()
        }