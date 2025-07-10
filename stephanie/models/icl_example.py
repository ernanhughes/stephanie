from datetime import datetime

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.ext.declarative import declarative_base

from stephanie.models.base import Base


class ICLExampleORM(Base):
    __tablename__ = "worldview_icl_examples"

    id = Column(Integer, primary_key=True)
    worldview_id = Column(Integer, ForeignKey("worldviews.id"))
    prompt = Column(Text)
    response = Column(Text)
    task_type = Column(String)
    score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
