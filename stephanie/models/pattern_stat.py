# stephanie/models/pattern_stat.py
# models/pattern_stat.py
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String

from stephanie.models.base import Base


class PatternStatORM(Base):
    __tablename__ = "cot_pattern_stats"

    id = Column(Integer, primary_key=True)

    # Foreign keys
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False)
    hypothesis_id = Column(Integer, ForeignKey("hypotheses.id"), nullable=False)

    # Agent/Model Info
    model_name = Column(String, nullable=False)
    agent_name = Column(String, nullable=False)

    # Rubric Dimension + Label
    dimension = Column(String, nullable=False)  # e.g., "Inference Style"
    label = Column(String, nullable=False)  # e.g., "Deductive"
    confidence_score = Column(Float)  # Optional numeric score

    # Timestamps
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
