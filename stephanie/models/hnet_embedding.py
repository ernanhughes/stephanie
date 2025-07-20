# stephanie/models/embedding.py

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Integer, Text
from sqlalchemy.sql import func

from stephanie.models.base import Base


class HNetEmbeddingORM(Base):
    __tablename__ = "hnet_embeddings"

    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=True)
    embedding = Column(Vector(1024), nullable=True)
    created_at = Column(DateTime, default=func.now())
    text_hash = Column(Text, nullable=True)

    def __repr__(self):
        return f"<HNetEmbeddingORM(id={self.id}, text_hash={self.text_hash[:10]}...)>"
