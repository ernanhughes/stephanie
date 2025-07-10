# stephanie/models/worldview.py

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, Text

from stephanie.models.base import Base


class WorldviewORM(Base):
    __tablename__ = "worldviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False)
    description = Column(Text)
    goal = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_data = Column(JSON) 
    db_path = Column(Text)
    active = Column(Boolean, default=True)

    def to_dict(self):
        return {
            "id": self.id,
            "goal": self.goal,
            "name": self.name,
            "db_path": self.db_path,
            "created_at": self.created_at.isoformat(),
            "active": self.active
        }
