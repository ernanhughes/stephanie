# stephanie/models/measurement.py
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class MeasurementORM(Base):
    __tablename__ = 'measurements'

    id = Column(Integer, primary_key=True)
    
    # Target entity being measured
    entity_type = Column(Text, nullable=False)  # e.g., "Cartridge", "Theorem", "Prompt"
    entity_id = Column(Integer, nullable=False)  # ID of the measured object
    
    # Type of measurement
    metric_name = Column(Text, nullable=False)  # e.g., "domain_density", "semantic_coverage"
    
    # Value(s) of the measurement
    value = Column(JSON)  # Could be float, list, dict, etc.
    
    # Contextual metadata
    context = Column(JSON)  # Optional: goal_id, session_id, etc.
    
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Measurement {self.metric_name} on {self.entity_type}[{self.entity_id}]>"