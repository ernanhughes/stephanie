# models/reflection_delta.py
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String)
from sqlalchemy.orm import relationship

from .base import Base


class ReflectionDeltaORM(Base):
    __tablename__ = "reflection_deltas"

    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey("goals.id"))
    run_id_a = Column(String, nullable=False)
    run_id_b = Column(String, nullable=False)
    score_a = Column(Float)
    score_b = Column(Float)
    score_delta = Column(Float)
    pipeline_a = Column(JSON)
    pipeline_b = Column(JSON)
    pipeline_diff = Column(JSON)
    strategy_diff = Column(Boolean, default=False)
    model_diff = Column(Boolean, default=False)
    rationale_diff = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    goal = relationship("GoalORM", back_populates="reflection_deltas")