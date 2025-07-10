# models/node_orm.py ---

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base

from stephanie.models.base import Base


class NodeORM(Base):
    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True)
    goal_id = Column(String, index=True)
    pipeline_run_id = Column(Integer, index=True)
    stage_name = Column(String, index=True)
    config = Column(JSON)
    hypothesis = Column(String)  # output or intermediate result
    metric = Column(Float)
    valid = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "pipeline_run_id": self.pipeline_run_id,
            "stage_name": self.stage_name,
            "config": self.config,
            "hypothesis": self.hypothesis,
            "metric": self.metric,
            "valid": self.valid,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    