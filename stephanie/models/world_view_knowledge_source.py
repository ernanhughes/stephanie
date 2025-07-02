from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

from stephanie.models.base import Base

class WorldviewKnowledgeSourceORM(Base):
    __tablename__ = "worldview_knowledge_sources"

    id = Column(Integer, primary_key=True)
    worldview_id = Column(Integer, ForeignKey("worldviews.id"))
    source_type = Column(String)  # e.g., "document", "agent", "ICLExample"
    source_id = Column(String)    # e.g., document hash or UUID
    metadata = Column(JSON)
