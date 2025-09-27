# stephanie/models/selfplay_item.py
from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB

from stephanie.models.base import Base
from stephanie.utils.date_utils import utcnow

# If you're on non-Postgres, swap JSONB for Text and stringify in the store.
JSONType = JSONB


class SelfPlayItemORM(Base):
    __tablename__ = "selfplay_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Partition key for grouping (e.g., "casebook:123", "paper:abc")
    scope = Column(Text, nullable=False, index=True)
    # Logical stream within a scope: "pool", "beam", "winners", "metrics", or anything you like
    buffer_name = Column(Text, nullable=False, index=True)

    # Milliseconds since epoch for stable ordering (don’t depend on db clock skew)
    ts_ms = Column(BigInteger, nullable=False, index=True)

    # Optional created_at (server time) for human debugging
    created_at = Column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )

    # Arbitrary payload; keep it tight (scores, text refs, meta, etc.)
    data = Column(JSONType, nullable=False)

    __table_args__ = (
        Index(
            "ix_selfplay_scope_buf_ts",
            "scope",
            "buffer_name",
            "ts_ms",
            postgresql_using="btree",
        ),
    )

    def __repr__(self):
        return f"<SelfPlayItemORM id={self.id} scope={self.scope} buffer={self.buffer_name} ts_ms={self.ts_ms}>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scope": self.scope,
            "buffer_name": self.buffer_name,
            "ts_ms": self.ts_ms,
            "created_at": self.created_at.isoformat()
            if self.created_at
            else None,
            "data": self.data,
        }
