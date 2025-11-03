# stephanie/models/report.py
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class ReportORM(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    goal = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    path = Column(String, nullable=True)
    content = Column(Text, nullable=True)   # optional: full markdown content
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)

    # Relationship back to pipeline run
    pipeline_run = relationship("PipelineRunORM", back_populates="reports")

    def __repr__(self):
        return f"<Report id={self.id} run_id={self.run_id} goal='{self.goal}'>"

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "goal": self.goal,
            "summary": self.summary,
            "path": self.path,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
