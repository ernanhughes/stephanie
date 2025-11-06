# stephanie/models/world_view_evidence.py
from __future__ import annotations


from sqlalchemy import (Column, Float, ForeignKey, Integer,
                        String, Text)

from stephanie.models.base import Base


class WorldViewEvidenceORM(Base):
    __tablename__ = "worldview_evidence"

    id = Column(Integer, primary_key=True)
    worldview_id = Column(Integer, ForeignKey("worldviews.id"))
    hypothesis_id = Column(Integer, ForeignKey("worldview_hypotheses.id"))
    evidence_text = Column(Text)
    confidence = Column(Float)
    evidence_type = Column(String)  # "support" or "contradiction"
    source_id = Column(String)  # reference to knowledge source
