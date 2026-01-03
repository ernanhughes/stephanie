# stephanie/orm/document.py
from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from stephanie.orm.base import Base


class DocumentORM(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    source = Column(String, nullable=False)
    external_id = Column(String, nullable=True)
    domain_label = Column(String, nullable=True)
    url = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    text = Column(Text, nullable=True)
    goal_id = Column(
        Integer, ForeignKey("goals.id", ondelete="SET NULL"), nullable=True
    )
    date_added = Column(DateTime(timezone=True), server_default=func.now())

    domains = Column(ARRAY(String), nullable=True)

    embedding_id = Column(
        Integer, ForeignKey("embeddings.id", ondelete="SET NULL"), nullable=True
    )
    embedding = relationship("EmbeddingORM", foreign_keys=[embedding_id])

    sections = relationship(
        "DocumentSectionORM", back_populates="document", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "external_id": self.external_id,
            "url": self.url,
            "summary": self.summary,
            "text": self.text,
            "goal_id": self.goal_id,
            "embedding_id": self.embedding_id,
            "domains": self.domains,
            "date_added": self.date_added.isoformat() if self.date_added else None,
        }
