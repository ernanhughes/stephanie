# models/score.py

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class ScoreORM(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False)
    dimension = Column(String, nullable=False)
    score = Column(Float)
    weight = Column(Float)
    rationale = Column(Text)
    prompt_hash = Column(Text)
    source = Column(Text, default="llm")    

    evaluation = relationship("EvaluationORM", back_populates="dimension_scores")

    def to_dict(self):
        return {
            "id": self.id,
            "evaluation_id": self.evaluation_id,
            "dimension": self.dimension,
            "score": self.score,
            "source": self.source,
            "weight": self.weight,
            "rationale": self.rationale
        }

    def __repr__(self):
        return (
            f"<ScoreORM(id={self.id}, eval_id={self.evaluation_id}, "
            f"dim='{self.dimension}', score={self.score}, "
            f"weight={self.weight}, rationale='{self.rationale[:40]}...')>"
        )