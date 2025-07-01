# models/mrq_memory_entry.py
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, REAL

from stephanie.models.base import Base


class MRQMemoryEntryORM(Base):
    __tablename__ = "mrq_memory"

    id = Column(Integer, primary_key=True)
    goal = Column(Text, nullable=False)
    strategy = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    reward = Column(Float, nullable=False)

    # Optional: Use these if storing embeddings
    embedding = Column(ARRAY(REAL))  # Or use pgvector.ARRAY(Float)
    features = Column(JSON)   # Additional extracted features

    source = Column(String)   # e.g., manual, agent, refinement
    run_id = Column(String)
    metadata_ = Column("metadata", JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))