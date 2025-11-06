# stephanie/models/scorable_domain.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from stephanie.models.base import Base
from stephanie.utils.date_utils import iso_date  # optional, for consistency

class ScorableDomainORM(Base):
    __tablename__ = "scorable_domains"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scorable_id = Column(Integer, nullable=False)
    scorable_type = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "domain": self.domain,
            "score": self.score,
            "created_at": iso_date(self.created_at) if self.created_at else None,
        }

    def __repr__(self):
        return (f"<ScorableDomainORM(id={self.id}, "
                f"{self.scorable_type}:{self.scorable_id}, "
                f"domain={self.domain}, score={self.score})>")
