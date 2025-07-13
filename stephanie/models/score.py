# stephanie/models/score.py

import hashlib

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base
from stephanie.scoring.scorable import Scorable


class ScoreORM(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True)
    evaluation_id = Column(
        Integer, ForeignKey("evaluations.id", ondelete="CASCADE"), nullable=False
    )
    dimension = Column(String, nullable=False)
    score = Column(Float)
    energy = Column(Float)     
    uncertainty = Column(Float) 
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
            "rationale": self.rationale,
        }

    def __repr__(self):
        return (
            f"<ScoreORM(id={self.id}, eval_id={self.evaluation_id}, "
            f"dim='{self.dimension}', score={self.score}, "
            f"weight={self.weight}, rationale='{self.rationale[:40]}...')>"
        )

    @staticmethod
    def compute_prompt_hash(prompt: str, scorable: Scorable) -> str:
        """
        Compute a deterministic hash for a prompt, including scorable's id and target_type.
        This avoids collisions across different entity types that might share IDs or prompts.
        """
        raw = f"{prompt}|{scorable.id}|{scorable.target_type}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
