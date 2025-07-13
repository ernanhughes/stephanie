# stephanie/models/document_score.py
# models/document_score.py

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class DocumentScoreORM(Base):
    __tablename__ = "document_scores"

    id = Column(Integer, primary_key=True)
    evaluation_id = Column(
        Integer,
        ForeignKey("document_evaluations.id", ondelete="CASCADE"),
        nullable=False,
    )
    dimension = Column(String, nullable=False)
    score = Column(Float)
    weight = Column(Float)
    rationale = Column(Text)

    evaluation = relationship(
        "DocumentEvaluationORM", back_populates="dimension_scores"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "evaluation_id": self.evaluation_id,
            "dimension": self.dimension,
            "score": self.score,
            "weight": self.weight,
            "rationale": self.rationale,
        }
