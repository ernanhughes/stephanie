# co_ai/models/document.py

from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from co_ai.models.base import Base

class DocumentORM(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    source = Column(String, nullable=False)         # e.g., "arxiv", "huggingface", "github"
    external_id = Column(String, nullable=True)     # e.g., "2505.19590" (arXiv ID)
    url = Column(String, nullable=True)             # full paper URL
    content = Column(Text, nullable=True)           # full extracted paper text
    date_added = Column(DateTime(timezone=True), server_default=func.now())
