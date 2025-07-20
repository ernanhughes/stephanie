from datetime import datetime
from typing import Optional

from sqlalchemy import (JSON, Boolean, Column, DateTime, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class BeliefCartridgeORM(Base):
    __tablename__ = "belief_cartridges"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    source_id = Column(String, index=True)  # e.g., paper ID
    source_type = Column(String, index=True)  # e.g., "paper", "blog", "experiment"

    markdown_content = Column(Text, nullable=False)
    goal_tags = Column(JSON, default=list, nullable=False)  # Stored as JSON array
    domain_tags = Column(JSON, default=list, nullable=False)  # Stored as JSON array

    idea_payload = Column(JSON)  # Can be JSON blob or link to structured table
    rationale = Column(Text)
    is_active = Column(Boolean, default=True, index=True)

    # Provenance and application history
    derived_from = Column(JSON, default=list)  # List of belief IDs
    applied_in = Column(JSON, default=list)  # List of world IDs or run IDs
    version = Column(Integer, default=1)

    # Optional attachments
    memcube_id = Column(String, index=True)  # Reference to MemCube
    debug_log = Column(JSON)  # Dictionary of debug info

    # Relationships
    evaluations = relationship("EvaluationORM", back_populates="belief_cartridge", cascade="all, delete-orphan")

    def latest_score_dict(self) -> Optional[dict]:
        """
        Return the most recent EvaluationORM's dimension scores as a dictionary.
        """
        if not self.evaluations:
            return None

        latest = sorted(self.evaluations, key=lambda e: e.created_at)[-1]
        return {
            s.dimension: s.score for s in latest.dimension_scores
        }

    def epistemic_gain(self) -> float:
        scores = self.latest_score_dict()
        return scores.get("epistemic_gain", 0.0) if scores else 0.0 