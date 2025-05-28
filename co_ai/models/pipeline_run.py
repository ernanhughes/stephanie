# models/pipeline_run.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base

class PipelineRunORM(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, nullable=False)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False)
    pipeline = Column(JSON)  # Stored as JSONB or TEXT[]
    strategy = Column(String)
    model_name = Column(String)
    run_config = Column(JSON)
    lookahead_context = Column(JSON)
    symbolic_suggestion = Column(JSON)
    extra_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    goal = relationship("GoalORM", back_populates="pipeline_runs")
    hypotheses = relationship("HypothesisORM", back_populates="pipeline_run")