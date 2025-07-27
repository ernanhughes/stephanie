# stephanie/models/training_stats.py
from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class TrainingStatsORM(Base):
    __tablename__ = "training_stats"
    
    id = Column(Integer, primary_key=True)
    
    # Model identification
    model_type = Column(String, nullable=False)
    target_type = Column(String, nullable=False)
    dimension = Column(String, nullable=False)
    version = Column(String, nullable=False)
    embedding_type = Column(String, nullable=False)
    
    # Training metrics
    q_loss = Column(Float, nullable=True)
    v_loss = Column(Float, nullable=True)
    pi_loss = Column(Float, nullable=True)
    avg_q_loss = Column(Float, nullable=True)
    avg_v_loss = Column(Float, nullable=True)
    avg_pi_loss = Column(Float, nullable=True)
    
    # Policy metrics
    policy_entropy = Column(Float, nullable=True)
    policy_stability = Column(Float, nullable=True)
    policy_logits = Column(JSON, nullable=True)
    
    # Training configuration
    config = Column(JSON, nullable=True)
    
    # Dataset stats
    sample_count = Column(Integer, default=0)
    valid_samples = Column(Integer, default=0)
    invalid_samples = Column(Integer, default=0)
    
    # Timing
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    
    # Relationships
    goal_id = Column(Integer, ForeignKey("goals.id", ondelete="SET NULL"), nullable=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    
    # Back-populates
    goal = relationship("GoalORM", back_populates="training_stats")
    model_version = relationship("ModelVersionORM", back_populates="training_stats")
    
    def __repr__(self):
        return f"<TrainingStats(id={self.id}, dim='{self.dimension}', model='{self.model_type}', version='{self.version}', q_loss={self.avg_q_loss:.4f})>"
    
    @classmethod
    def from_dict(cls, stats: dict, **kwargs):
        """Create TrainingStatsORM from training results"""
        return cls(
            model_type=kwargs.get("model_type", "ebt"),
            target_type=kwargs.get("target_type", "document"),
            dimension=kwargs.get("dimension", "alignment"),
            version=kwargs.get("version", "v1"),
            embedding_type=kwargs.get("embedding_type", "hnet"),
            sample_count=kwargs.get("sample_count", 0),
            valid_samples=kwargs.get("valid_samples", 0),
            invalid_samples=kwargs.get("invalid_samples", 0),
            q_loss=stats.get("q_loss"),
            v_loss=stats.get("v_loss"),
            pi_loss=stats.get("pi_loss"),
            avg_q_loss=stats.get("avg_q_loss"),
            avg_v_loss=stats.get("avg_v_loss"),
            avg_pi_loss=stats.get("avg_pi_loss"),
            policy_entropy=stats.get("policy_entropy"),
            policy_stability=stats.get("policy_stability"),
            policy_logits=stats.get("policy_logits"),
            config=kwargs.get("config", {}),
            start_time=kwargs.get("start_time", datetime.utcnow())
        )