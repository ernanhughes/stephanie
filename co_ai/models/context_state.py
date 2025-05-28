# models/context_state.py
from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, Boolean
from datetime import datetime, timezone
from co_ai.models.base import Base


class ContextStateORM(Base):
    __tablename__ = "context_states"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, nullable=False)
    stage_name = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    is_current = Column(Boolean, default=True)
    context = Column(JSON, nullable=False)  # Stored as JSONB or TEXT
    preferences = Column(JSON)
    extra_data = Column(JSON)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))