from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Integer, Text

from stephanie.models.base import Base


class MRQPreferencePairORM(Base):
    __tablename__ = "mrq_preference_pairs"

    id = Column(Integer, primary_key=True)

    goal = Column(Text, nullable=False)
    prompt = Column(Text, nullable=False)

    output_a = Column(Text, nullable=False)
    output_b = Column(Text, nullable=False)
    preferred = Column(Text, nullable=False)  # 'a' or 'b'

    fmt_a = Column(Text)  # e.g., direct, short_cot, code, long_cot
    fmt_b = Column(Text)
    
    difficulty = Column(Text)
    
    features = Column(JSON)  # Optional: extra metadata
    run_id = Column(Text)
    source = Column(Text)  # e.g., arm_dataloader, user, 
    
    created_at = Column(DateTime, default=datetime.now(timezone.utc))


    def to_dict(self):
        return {
            "id": self.id,
            "goal": self.goal,
            "prompt": self.prompt,
            "output_a": self.output_a,
            "output_b": self.output_b,
            "preferred": self.preferred,
            "fmt_a": self.fmt_a,
            "fmt_b": self.fmt_b,
            "difficulty": self.difficulty,
            "features": self.features or {},
            "run_id": self.run_id,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
