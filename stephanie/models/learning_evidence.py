# stephanie/models/learning_evidence.py
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from datetime import datetime
from stephanie.models.base import Base

class LearningEvidenceORM(Base):
    __tablename__ = "learning_evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, nullable=False)
    strategy_version = Column(Integer, nullable=False)
    old_threshold = Column(Float, nullable=True)
    new_threshold = Column(Float, nullable=True)
    old_weights = Column(JSON, nullable=True)
    new_weights = Column(JSON, nullable=True)
    avg_gain = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    meta = Column(JSON, default={})


    def __repr__(self):
        return f"<LearningEvidenceORM(id={self.id}, doc_id={self.doc_id}, strategy_version={self.strategy_version})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "strategy_version": self.strategy_version,
            "old_threshold": self.old_threshold,
            "new_threshold": self.new_threshold,
            "old_weights": self.old_weights,
            "new_weights": self.new_weights,
            "avg_gain": self.avg_gain,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "meta": self.meta,
        }