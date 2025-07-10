from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

from stephanie.models.base import Base


class WorldviewORM(Base):
    __tablename__ = "worldviews"

    id = Column(Integer, primary_key=True)
    goal = Column(Text)  # The objective of this worldview
    core_thesis = Column(Text)  # A single paragraph distillation of the worldview
    knowledge_sources = Column(JSON)  # Source documents, examples, etc.
    icl_examples = Column(JSON)  # Used ICL examples and scores
    hypotheses = Column(JSON)  # Extracted or inferred hypotheses
    contradictions = Column(JSON)  # Known flaws, limits, edge cases
    supporting_evidence = Column(JSON)  # Key evidence backing core claims
    confidence_score = Column(Float)  # Aggregate confidence
    generation = Column(Integer, default=0)  # Evolution index
    created_at = Column(DateTime, default=datetime.utcnow)
