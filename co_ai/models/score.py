# models/score.py

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from co_ai.models.base import Base


class ScoreORM(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False)
    dimension = Column(String, nullable=False)
    score = Column(Float)
    weight = Column(Float)
    rationale = Column(Text)

    evaluation = relationship("EvaluationORM", back_populates="dimension_scores")

    def to_dict(self):
        return {
            "id": self.id,
            "evaluation_id": self.evaluation_id,
            "dimension": self.dimension,
            "score": self.score,
            "weight": self.weight,
            "rationale": self.rationale
        }
