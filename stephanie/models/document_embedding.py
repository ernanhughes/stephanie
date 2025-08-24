# stephanie/models/document_embedding.py

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from stephanie.models.base import Base


class DocumentEmbeddingORM(Base):
    __tablename__ = "document_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, nullable=False)         # string ID
    document_type = Column(String, nullable=False)       # polymorphic scorable type
    embedding_id = Column(Integer, nullable=False)       # FK into hnet/hf/llama embeddings
    embedding_type = Column(String, nullable=False)      # embedding backend
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return (
            f"<DocumentEmbeddingORM(doc={self.document_id}, "
            f"type={self.document_type}, emb_type={self.embedding_type}, emb_id={self.embedding_id})>"
        )
