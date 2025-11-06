# stephanie/models/scorable_entity.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, Integer, String, Text)

from stephanie.models.base import Base


class ScorableEntityORM(Base):
    __tablename__ = "scorable_entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scorable_id = Column(String, nullable=False, index=True)
    scorable_type = Column(String, nullable=False, index=True)
    entity_text = Column(Text, nullable=False)
    entity_text_norm = Column(String, nullable=False, index=True) 
    entity_type = Column(String, nullable=True)
    start = Column(Integer, nullable=True)
    end = Column(Integer, nullable=True)
    ner_confidence = Column(Float, nullable=True) 
    similarity = Column(Float, nullable=True)
    source_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "entity_text": self.entity_text,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "similarity": self.similarity,
            "source_text": self.source_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<ScorableEntityORM({self.scorable_type}:{self.scorable_id}, entity='{self.entity_text}', type={self.entity_type})>"
