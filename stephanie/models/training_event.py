# stephanie/models/training_event.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (BigInteger, Boolean, Column, DateTime, Float, Integer,
                        String, Text)
from sqlalchemy.dialects.postgresql import JSONB

from stephanie.models.base import Base


class TrainingEventORM(Base):
    __tablename__ = "training_events"

    id           = Column(BigInteger, primary_key=True, autoincrement=True)
    model_key    = Column(String, nullable=False)
    dimension    = Column(String, nullable=False)
    goal_id      = Column(String, nullable=True)
    pipeline_run_id = Column(Integer, nullable=True)
    agent_name   = Column(String, nullable=True)

    kind         = Column(String, nullable=False)  # 'pairwise' | 'pointwise'

    query_text   = Column(Text, nullable=True)
    pos_text     = Column(Text, nullable=True)
    neg_text     = Column(Text, nullable=True)

    cand_text    = Column(Text, nullable=True)
    label        = Column(Integer, nullable=True)

    weight       = Column(Float, nullable=False, default=1.0)
    trust        = Column(Float, nullable=False, default=0.0)
    source       = Column(String, nullable=False, default="memento")
    meta         = Column(JSONB, nullable=False, default=dict)

    fp           = Column(String(40), nullable=True, unique=True)
    processed    = Column(Boolean, nullable=False, default=False)
    created_at   = Column(DateTime, nullable=False, default=datetime.now)

    def __repr__(self):
        return f"<TrainingEvent(id={self.id}, model_key={self.model_key}, dimension={self.dimension}, kind={self.kind}, created_at={self.created_at})>"

    def to_dict(self):
        return {
            "id": self.id,
            "model_key": self.model_key,
            "dimension": self.dimension,
            "goal_id": self.goal_id,
            "pipeline_run_id": self.pipeline_run_id,
            "agent_name": self.agent_name,
            "kind": self.kind,
            "query_text": self.query_text,
            "pos_text": self.pos_text,
            "neg_text": self.neg_text,
            "cand_text": self.cand_text,
            "label": self.label,
            "weight": self.weight,
            "trust": self.trust,
            "source": self.source,
            "meta": self.meta,
            "fp": self.fp,
            "processed": self.processed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }