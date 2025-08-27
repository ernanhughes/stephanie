# stephanie/models/scorable_domain.py
from sqlalchemy import Column, Float, Integer, String, UniqueConstraint, DateTime
from datetime import datetime

from stephanie.models.base import Base


class ScorableDomainORM(Base):
    __tablename__ = "scorable_domains"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to Scorable (document, plan_trace, prompt, etc.)
    scorable_id = Column(String, nullable=False, index=True)   # maps to Scorable.id
    scorable_type = Column(String, nullable=False, index=True) # maps to Scorable.target_type

    # Domain classification
    domain = Column(String, nullable=False, index=True)
    score = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("scorable_id", "scorable_type", "domain", name="uq_scorable_domain"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "domain": self.domain,
            "score": self.score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<ScorableDomainORM(id={self.id}, scorable_id={self.scorable_id}, scorable_type={self.scorable_type}, domain={self.domain}, score={self.score})>"
