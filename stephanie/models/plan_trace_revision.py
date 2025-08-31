# stephanie/models/plan_trace_revision.py
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class PlanTraceRevisionORM(Base):
    __tablename__ = "plan_trace_revisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_trace_id = Column(
        String, ForeignKey("plan_traces.trace_id", ondelete="CASCADE"), nullable=False
    )

    revision_type = Column(String, nullable=False)   # e.g. "correction", "improvement", "feedback"
    revision_text = Column(Text, nullable=False)     # free-form feedback
    source = Column(String, nullable=True)           # "user", "llm", "scorer", etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    plan_trace = relationship("PlanTraceORM", back_populates="revisions")

    def to_dict(self):
        return {
            "id": self.id,
            "plan_trace_id": self.plan_trace_id,
            "revision_type": self.revision_type,
            "revision_text": self.revision_text,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }
