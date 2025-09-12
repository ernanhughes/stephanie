from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer

from stephanie.models.base import Base


class EntityCacheORM(Base):
    __tablename__ = "entity_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # reference to canonical embedding
    embedding_ref = Column(Integer, ForeignKey("scorable_embeddings.id"), nullable=False)

    # cached KG/NER results (neighbors, metadata, etc.)
    results_json = Column(JSON, nullable=True)

    # timestamp of last refresh
    last_updated = Column(DateTime, default=datetime.now, nullable=False)

    def __repr__(self):
        return f"<EntityCacheORM embedding_ref={self.embedding_ref} updated={self.last_updated}>"

    def to_dict(self):
        return {
            "id": self.id,
            "embedding_ref": self.embedding_ref,
            "results_json": self.results_json,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }
