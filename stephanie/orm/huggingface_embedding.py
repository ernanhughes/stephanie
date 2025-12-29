# stephanie/orm/embedding.py
from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Integer, Text
from sqlalchemy.sql import func

from stephanie.orm.base import Base


class HuggingfaceEmbeddingORM(Base):
    __tablename__ = "huggingface_embeddings"

    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=True)
    embedding = Column(Vector(1024), nullable=True)
    created_at = Column(DateTime, default=func.now())
    text_hash = Column(Text, nullable=True)

    def __repr__(self):
        return f"<HNetHuggingfaceEmbeddingORM(id={self.id}, text_hash={self.text_hash[:10]}...)>"
