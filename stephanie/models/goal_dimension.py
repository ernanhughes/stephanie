from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class GoalDimensionORM(Base):
    __tablename__ = "goal_dimensions"

    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey("goals.id", ondelete="CASCADE"), nullable=False)
    dimension = Column(String, nullable=False)
    rank = Column(Integer, default=0)
    source = Column(String, default="llm")
    similarity_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    goal = relationship("GoalORM", back_populates="dimensions")

    def to_dict(self):
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "dimension": self.dimension,
            "rank": self.rank,
            "source": self.source,
            "similarity_score": self.similarity_score,
            "created_at": self.created_at.isoformat(),
        }
    
    def __repr__(self):
        return f"<GoalDimensionORM(id={self.id}, dimension='{self.dimension}', goal_id={self.goal_id})>"