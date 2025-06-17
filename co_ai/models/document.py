# co_ai/models/document.py

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from co_ai.models.base import Base


class DocumentORM(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    source = Column(String, nullable=False)         # e.g., "arxiv", "huggingface", "github"
    external_id = Column(String, nullable=True)     # e.g., "2505.19590" (arXiv ID) oh
    domain_label = Column(String, nullable=True)
    url = Column(String, nullable=True)             # full paper_score URL
    summary = Column(Text, nullable=True)           # optional abstract or extracted summary
    content = Column(Text, nullable=True)           # full extracted paper_score text
    goal_id = Column(Integer, ForeignKey("goals.id", ondelete="SET NULL"), nullable=True)  # optional link
    date_added = Column(DateTime(timezone=True), server_default=func.now())

    domains = relationship("DocumentDomainORM", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "external_id": self.external_id,
            "url": self.url,
            "summary": self.summary,
            "content": self.content,
            "goal_id": self.goal_id,
            "date_added": self.date_added.isoformat() if self.date_added else None
        }
