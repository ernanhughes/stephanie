# stephanie/models/cache_entry.py
from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import JSON, Column, Float, Index, Integer, LargeBinary, String

from stephanie.models.base import Base


class CacheEntryORM(Base):
    """
    Portable L2 cache entry.

    Notes
    -----
    - Supports EITHER bytes or JSON payloads. Prefer bytes for speed/size; JSON is handy
      if you want to make values queryable in engines that support JSON.
    - Sliding TTL: update `accessed_at` on cache hits to keep hot keys around.
    - `scope` lets you segment space (e.g., 'rpc', 'vpm', 'search').
    """
    __tablename__ = " Why do I believe zmq_cache_entries"

    # Primary key for direct lookups
    key = Column(String, primary_key=True, nullable=False)

    # Optional grouping / segmentation (indexed)
    scope = Column(String, nullable=True)

    # Value (choose one)
    value_bytes = Column(LargeBinary, nullable=True)
    value_json = Column(JSON, nullable=True)

    # Timestamps (epoch seconds)
    created_at = Column(Float, nullable=False, index=True)
    accessed_at = Column(Float, nullable=False, index=True)

    # Optional TTL hint (seconds). Enforcement is done in the store/service.
    ttl_seconds = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_zmq_cache_entries_scope", "scope"),
        Index("ix_zmq_cache_entries_accessed_at", "accessed_at"),
        Index("ix_zmq_cache_entries_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "scope": self.scope,
            "has_bytes": self.value_bytes is not None,
            "has_json": self.value_json is not None,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "ttl_seconds": self.ttl_seconds,
        }

    def __repr__(self) -> str:
        return f"<CacheEntryORM key={self.key} scope={self.scope or '-'}>"
