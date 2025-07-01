
# models/comparison_preference.py
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text

from stephanie.models.base import Base


class ComparisonPreferenceORM(Base):
    __tablename__ = "comparison_preferences"
    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer)
    preferred_tag = Column(String)
    rejected_tag = Column(String)
    preferred_run_id = Column(Integer)
    rejected_run_id = Column(Integer)
    preferred_score = Column(Float)
    rejected_score = Column(Float)
    dimension_scores = Column(JSON)
    reason = Column(Text)
    source = Column(String, default="pipeline_comparison")
    created_at = Column(DateTime, default=datetime.utcnow)
