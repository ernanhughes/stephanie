from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class BeliefTuneLogORM(Base):
    __tablename__ = "belief_tune_log"

    id = Column(Integer, primary_key=True)
    belief_id = Column(Integer, ForeignKey("belief.id"), index=True)
    old_score = Column(Float)
    new_score = Column(Float)
    source = Column(String)  # e.g. "external", "rival_eval", "user_feedback"
    rationale = Column(String)
    tuned_at = Column(DateTime, default=datetime.utcnow)

    belief = relationship("BeliefORM", back_populates="tune_logs")
