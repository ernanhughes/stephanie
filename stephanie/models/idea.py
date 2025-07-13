# stephanie/models/idea.py
# models/idea.py
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class IdeaORM(Base):
    __tablename__ = "ideas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    idea_text = Column(String, nullable=False)
    parent_goal = Column(String)
    focus_area = Column(String)
    strategy = Column(String)
    source = Column(String)
    origin = Column(String)
    extra_data = Column(JSON)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to GoalORM
    goal = relationship("GoalORM", back_populates="ideas")

    def to_dict(self):
        return {
            "id": self.id,
            "idea_text": self.idea_text,
            "parent_goal": self.parent_goal,
            "focus_area": self.focus_area,
            "strategy": self.strategy,
            "source": self.source,
            "origin": self.origin,
            "extra_data": self.extra_data or {},  # Avoid NoneType issues
            "goal_id": self.goal_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
