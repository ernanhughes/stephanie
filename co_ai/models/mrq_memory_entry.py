# models/mrq_memory_entry.py
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON
from datetime import datetime
from co_ai.models.base import Base
from sqlalchemy.dialects.postgresql import ARRAY, REAL

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
    created_at = Column(DateTime, default=datetime.utcnow)