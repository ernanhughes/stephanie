# stephanie/models/score_attribute.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from stephanie.models.base import Base

class ScoreAttributeORM(Base):
    __tablename__ = "score_attributes"
    
    id = Column(Integer, primary_key=True)
    score_id = Column(Integer, ForeignKey("scores.id", ondelete="CASCADE"), nullable=False)
    key = Column(String(64), nullable=False)
    value = Column(Text, nullable=False)
    data_type = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to ScoreORM
    score = relationship("ScoreORM", back_populates="attributes")
    
    def __repr__(self):
        return f"<ScoreAttributeORM(score_id={self.score_id}, key='{self.key}', type='{self.data_type}')>"