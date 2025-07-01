# models/sharpening_prediction.py
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class SharpeningPredictionORM(Base):
    __tablename__ = "sharpening_predictions"

    id = Column(Integer, primary_key=True)

    # Goal context
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False)

    # Prompt that led to the comparison
    prompt_text = Column(String, nullable=False)

    # Original and evolved outputs
    output_a = Column(String, nullable=False)
    output_b = Column(String, nullable=False)

    # Evaluation results
    preferred = Column(String)  # 'a' or 'b'
    predicted = Column(String)  # 'a' or 'b'

    value_a = Column(Float, nullable=False)
    value_b = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    
    # Relationships
    goal = relationship("GoalORM", back_populates="sharpening_predictions")

    def to_dict(self, include_relationships: bool = False) -> dict:
        """
        Convert this ORM instance to a dictionary representation.

        Args:
            include_relationships (bool): Whether to include related objects like GoalORM

        Returns:
            dict: Dictionary of all fields
        """
        result = {
            "id": self.id,
            "goal_id": self.goal_id,
            "prompt_text": self.prompt_text,
            "output_a": self.output_a,
            "output_b": self.output_b,
            "preferred": self.preferred,
            "predicted": self.predicted,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

        if include_relationships:
            result["goal"] = self.goal.to_dict() if self.goal else None

        return result