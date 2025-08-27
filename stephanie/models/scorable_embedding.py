# stephanie/models/document_embedding.py

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from stephanie.models.base import Base


class ScorableEmbeddingORM(Base):
    __tablename__ = "scorable_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scorable_id = Column(String, nullable=False)         # string ID
    scorable_type = Column(String, nullable=False)       # polymorphic scorable type
    embedding_id = Column(Integer, nullable=False)       # FK into hnet/hf/llama embeddings
    embedding_type = Column(String, nullable=False)      # embedding backend
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return (
            f"<ScorableEmbeddingORM(doc={self.scorable_id}, "
            f"type={self.scorable_type}, emb_type={self.embedding_type}, emb_id={self.embedding_id})>"
        )
