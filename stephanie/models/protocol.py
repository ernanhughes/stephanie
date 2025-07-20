# stephanie/models/protocol_orm.py
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, String
from sqlalchemy.dialects.postgresql import JSONB

from stephanie.models.base import Base


class ProtocolORM(Base):
    __tablename__ = 'protocols'

    # Unique identifier
    name = Column(String, primary_key=True)

    # Human-readable info
    description = Column(String, nullable=True)
    capability = Column(String, index=True)  # e.g., "question_answering", "scoring"

    # Structural metadata
    input_format = Column(JSONB, nullable=True)     # {key: type}
    output_format = Column(JSONB, nullable=True)
    failure_modes = Column(JSONB, nullable=True)    # ["timeout", "hallucination"]
    depends_on = Column(JSONB, nullable=True)       # e.g., ["embedding", "search"]
    tags = Column(JSONB, nullable=True)             # e.g., ["llm", "fast", "cot"]

    # Preference modeling (used by selector)
    preferred_for = Column(JSONB, nullable=True)    # e.g., ["short_tasks", "factual"]
    avoid_for = Column(JSONB, nullable=True)        # e.g., ["slow", "long_outputs"]

    # Evaluation / learning metadata
    average_score = Column(Float, nullable=True)    # Computed by MRQ, etc.
    use_count = Column(Float, default=0.0)          # How often this was selected
    last_used = Column(DateTime, nullable=True)
    disabled = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Protocol(name={self.name}, capability={self.capability}, tags={self.tags})>"
