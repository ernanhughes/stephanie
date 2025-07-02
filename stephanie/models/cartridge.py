# models/cartridge.py
from datetime import datetime
from sqlalchemy import (Column, DateTime, ForeignKey, Integer, String)

from stephanie.models.base import Base

class CartridgeORM(Base):
    __tablename__ = 'cartridges'
    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey('goals.id'))
    source_type = Column(String)
    source_uri = Column(String)
    markdown_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
