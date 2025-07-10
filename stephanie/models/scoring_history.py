from sqlalchemy import Column, Integer, String, Float, Boolean, Text, JSON, TIMESTAMP, ForeignKey
from datetime import datetime

from stephanie.models.base import Base


class ScoringHistoryORM(Base):
    __tablename__ = "scoring_history"

    id = Column(Integer, primary_key=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    goal_id = Column(Integer)
    target_id = Column(Integer, nullable=False)
    target_type = Column(Text, nullable=False)
    dimension = Column(Text, nullable=False)
    raw_score = Column(Float)
    transformed_score = Column(Float)
    uncertainty_score = Column(Float)
    method = Column(Text, nullable=False)
    source = Column(Text)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    