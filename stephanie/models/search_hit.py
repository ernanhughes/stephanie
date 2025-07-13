# stephanie/models/search_hit.py

from sqlalchemy import JSON, Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class SearchHitORM(Base):
    __tablename__ = "search_hits"

    id = Column(Integer, primary_key=True)
    query = Column(String, nullable=False)
    source = Column(String, nullable=False)
    result_type = Column(String, default="unknown")
    title = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    url = Column(String, nullable=True)

    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=True)
    parent_goal = Column(Text, nullable=True)
    strategy = Column(String, nullable=True)
    focus_area = Column(String, nullable=True)
    extra_data = Column(JSON, nullable=True)

    # Optional: relationship if you're linking this to a GoalORM
    goal = relationship("GoalORM", back_populates="search_hits", lazy="joined")

    def to_dict(self):
        return {
            "id": self.id,
            "query": self.query,
            "source": self.source,
            "result_type": self.result_type,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "goal_id": self.goal_id,
            "parent_goal": self.parent_goal,
            "strategy": self.strategy,
            "focus_area": self.focus_area,
            "extra_data": self.extra_data,
        }

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)
