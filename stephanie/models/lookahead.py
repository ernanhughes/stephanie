# models/lookahead.py
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class LookaheadORM(Base):
    __tablename__ = "lookaheads"

    id = Column(Integer, primary_key=True)

    # Goal info
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False)

    # Agent metadata
    agent_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)

    # Pipeline information
    input_pipeline = Column(String)  # Store as JSON string if needed
    suggested_pipeline = Column(String)  # Same here
    rationale = Column(Text)
    reflection = Column(Text)
    backup_plans = Column(Text)  # List[str] stored as JSON or newline-separated
    extra_data = Column("metadata", Text)  # Renamed to avoid conflict with SQLAlchemy
    run_id = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)

    goal = relationship("GoalORM", back_populates="lookaheads")
