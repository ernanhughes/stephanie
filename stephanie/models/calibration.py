# stephanie/models/calibration.py
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String

from stephanie.models.base import Base


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
    timestamp = Column(DateTime, default=datetime.utcnow)

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