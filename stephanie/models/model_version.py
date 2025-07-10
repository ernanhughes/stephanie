from sqlalchemy import Column, Integer, String, Float, Boolean, Text, JSON, TIMESTAMP, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

from stephanie.models.base import Base

class ModelVersionORM(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    model_type = Column(Text, nullable=False)
    target_type = Column(Text, nullable=False)
    dimension = Column(Text, nullable=False)
    version = Column(Text, nullable=False)
    trained_on = Column(JSON)
    performance = Column(JSON)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    active = Column(Boolean, default=True)
    extra_data = Column(JSON)
    model_path = Column(Text, nullable=False)
    encoder_path = Column(Text, nullable=True)
    tuner_path = Column(Text, nullable=True)
    scaler_path = Column(Text, nullable=True)
    meta_path = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    source = Column(Text, nullable=True)


    def __repr__(self):
        return f"<ModelVersionORM(model_type={self.model_type}, target_type={self.target_type}, dimension={self.dimension}, version={self.version})>"

    def to_dict(self):
        return {
            "id": self.id,
            "model_type": self.model_type,
            "target_type": self.target_type,
            "dimension": self.dimension,
            "version": self.version,
            "trained_on": self.trained_on,
            "performance": self.performance,
            "created_at": self.created_at.isoformat(),
            "active": self.active,
            "extra_data": self.extra_data,
            "model_path": self.model_path,
            "encoder_path": self.encoder_path,
            "tuner_path": self.tuner_path,
            "scaler_path": self.scaler_path,
            "meta_path": self.meta_path,
            "description": self.description,
            "source": self.source
        }