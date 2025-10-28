# stephanie/models/belief.py
from __future__ import annotations

from __future__ import annotations
from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer,
                        String, Text)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class BeliefORM(Base):
    __tablename__ = "beliefs"
    id = Column(Integer, primary_key=True)
    worldview_id = Column(Integer, ForeignKey("worldviews.id"))
    cartridge_id = Column(Integer, ForeignKey("cartridges.id"), nullable=True)
    summary = Column(Text)
    text = Column(Text)
    rationale = Column(Text)
    utility_score = Column(Float)
    novelty_score = Column(Float)
    domain = Column(String)
    status = Column(String, default="active")  # active | deprecated | pending_review
    created_at = Column(DateTime, default=datetime.now)
