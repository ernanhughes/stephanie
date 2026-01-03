# stephanie/orm/memcube.py
from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import JSON, TIMESTAMP, Column, Float, Integer, String, Text
from sqlalchemy.sql import func

from stephanie.orm.base import Base


class MemCubeORM(Base):
    __tablename__ = "memcubes"

    id = Column(String, primary_key=True)                    # hash-based ID + version
    scorable_id = Column(Integer, nullable=False)            # bigint → Integer in Python
    scorable_type = Column(String, nullable=False)           # e.g., 'document', 'theorem'
    content = Column(Text, nullable=False)                   # Raw text from Scorable
    dimension = Column(String, nullable=True)                # e.g., 'relevance', 'clarity'
    original_score = Column(Float, nullable=True)            # Pre-refinement score
    refined_score = Column(Float, nullable=True)             # Post-refinement score
    refined_content = Column(Text, nullable=True)            # Optional refined text
    version = Column(String, nullable=False)                 # e.g., 'v1', 'v2'
    source = Column(String, nullable=True)                   # e.g., 'user_input', 'inference_engine'
    model = Column(String, nullable=True)                    # e.g., 'gpt-4', 'llama3'
    priority = Column(Integer, default=5)                    # 1–10 scale
    sensitivity = Column(String, default='public')           # security tag
    ttl = Column(Integer, nullable=True)                     # Time-to-live in days
    usage_count = Column(Integer, default=0)
    extra_data = Column(JSON, default={})                    # Flexible metadata
    created_at = Column(TIMESTAMP, default=func.now())       # Use func.now() for server-side default
    last_modified = Column(TIMESTAMP, default=func.now(), onupdate=func.now())

    def to_dict(self, include_extra: bool = True) -> Dict[str, Any]:
        """
        Serialize MemCube to dictionary.
        Optionally exclude extra_data for lighter payloads.
        """
        data = {
            "id": self.id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "content": self.content,
            "dimension": self.dimension,
            "original_score": self.original_score,
            "refined_score": self.refined_score,
            "refined_content": self.refined_content,
            "version": self.version,
            "source": self.source,
            "model": self.model,
            "priority": self.priority,
            "sensitivity": self.sensitivity,
            "ttl": self.ttl,
            "usage_count": self.usage_count,
        }
        if include_extra:
            data["extra_data"] = self.extra_data or {}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemCubeORM":
        """
        Factory method to create MemCubeORM from dict (e.g., from API or config).
        Handles optional fields and defaults.
        """
        return cls(
            id=data["id"],
            scorable_id=data["scorable_id"],
            scorable_type=data["scorable_type"],
            content=data["content"],
            dimension=data.get("dimension"),
            original_score=data.get("original_score"),
            refined_score=data.get("refined_score"),
            refined_content=data.get("refined_content"),
            version=data["version"],
            source=data.get("source"),
            model=data.get("model"),
            priority=data.get("priority", 5),
            sensitivity=data.get("sensitivity", "public"),
            ttl=data.get("ttl"),
            usage_count=data.get("usage_count", 0),
            extra_data=data.get("extra_data", {}),
            created_at=data.get("created_at"),
            last_modified=data.get("last_modified"),
        )

    def __repr__(self):
        return (
            f"<MemCube(id={self.id}, "
            f"type={self.scorable_type}, "
            f"dim={self.dimension}, "
            f"score={self.refined_score or self.original_score})>"
        )