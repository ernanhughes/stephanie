# stephanie/models/context_state.py

from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, ForeignKey, Integer,
                        String)

from stephanie.models.base import Base


class ContextStateORM(Base):
    __tablename__ = "context_states"

    # Foreign key references
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=True)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=True)

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False)
    stage_name = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    is_current = Column(Boolean, default=True)
    context = Column(JSON, nullable=False)  # Stored as JSONB or TEXT
    trace = Column(JSON)
    preferences = Column(JSON)
    token_count = Column(Integer)
    extra_data = Column(JSON)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
