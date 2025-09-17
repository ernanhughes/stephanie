# stephanie/models/plan_trace_reuse_link.py
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class PlanTraceReuseLinkORM(Base):
    __tablename__ = "plan_trace_reuse_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parent_trace_id = Column(String, ForeignKey("plan_traces.trace_id"), nullable=False)
    child_trace_id = Column(String, ForeignKey("plan_traces.trace_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    parent_trace = relationship("PlanTraceORM", foreign_keys=[parent_trace_id])
    child_trace = relationship("PlanTraceORM", foreign_keys=[child_trace_id])

    def to_dict(self):
        return {
            "id": self.id,
            "parent_trace_id": self.parent_trace_id,
            "child_trace_id": self.child_trace_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
