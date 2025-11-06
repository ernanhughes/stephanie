# stephanie/models/scorable_embedding.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from stephanie.models.base import Base
from stephanie.utils.date_utils import iso_date


class ScorableEmbeddingORM(Base): 
    __tablename__ = "scorable_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Polymorphic owner of the embedding record
    scorable_id = Column(String, nullable=False)      # e.g., source_uri or internal id
    scorable_type = Column(String, nullable=False)    # e.g., 'document', 'hypothesis', 'cartridge'

    # Backend embedding reference (id inside your HNet/HF/Llama stores)
    embedding_id = Column(Integer, nullable=False)
    embedding_type = Column(String, nullable=False)               # e.g., 'hnet', 'hf', 'ollama'

    created_at = Column(DateTime, default=datetime.now, nullable=False)

    def __repr__(self):
        return (
            f"<ScorableEmbeddingORM(scorable={self.scorable_type}:{self.scorable_id}, "
            f"emb={self.embedding_type}:{self.embedding_id})>"
        )

    def to_dict(self) -> dict:
        """Serialize the ORM object into a dictionary."""
        return {
            "id": self.id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "embedding_id": self.embedding_id,
            "embedding_type": self.embedding_type,
            "created_at": iso_date(self.created_at) if self.created_at else None,

        }
