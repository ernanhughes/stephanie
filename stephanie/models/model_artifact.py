from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, UniqueConstraint, Index
from stephanie.models.base import Base

class ModelArtifactORM(Base):
    __tablename__ = "model_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)          # e.g., "mrq_knowledge_reward_head"
    version = Column(Integer, nullable=False, default=1)  # auto-incremented per name
    path = Column(String, nullable=False)          # filesystem path to the saved model
    tag = Column(String, nullable=True)            # optional: "prod", "staging", "exp-42"
    meta = Column(JSON, default={})                # arbitrary JSON (metrics, dims, etc.)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, nullable=False)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_artifacts_name_version"),
        Index("ix_model_artifacts_name", "name"),
        Index("ix_model_artifacts_tag", "tag"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "tag": self.tag,
            "meta": self.meta or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
