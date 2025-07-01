from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, String, Text

from stephanie.models.base import \
    Base  # Adjust this import to match your actual Base


class SharpeningResultORM(Base):
    __tablename__ = 'sharpening_results'

    id = Column(String, primary_key=True)  # Optional: Add UUID or auto-increment ID
    goal = Column(Text, nullable=False)
    prompt = Column(Text, nullable=False)
    template = Column(Text, nullable=False)
    original_output = Column(Text, nullable=False)
    sharpened_output = Column(Text, nullable=False)
    preferred_output = Column(Text, nullable=False)
    winner = Column(String, nullable=False)
    improved = Column(Boolean, nullable=False)
    comparison = Column(Text, nullable=False)
    score_a = Column(Float, nullable=False)
    score_b = Column(Float, nullable=False)
    score_diff = Column(Float, nullable=False)
    best_score = Column(Float, nullable=False)
    prompt_template = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self, include_nulls=False):
        data = {
            "id": self.id,
            "goal": self.goal,
            "prompt": self.prompt,
            "template": self.template,
            "original_output": self.original_output,
            "sharpened_output": self.sharpened_output,
            "preferred_output": self.preferred_output,
            "winner": self.winner,
            "improved": self.improved,
            "comparison": self.comparison,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "score_diff": self.score_diff,
            "best_score": self.best_score,
            "prompt_template": self.prompt_template,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if not include_nulls:
            data = {k: v for k, v in data.items() if v is not None}
        return data