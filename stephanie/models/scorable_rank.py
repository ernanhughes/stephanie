# stephanie/models/scorable_rank.py
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stephanie.models.base import Base


class ScorableRankORM(Base):
    """
    Stores ranking results for a single scorable under a query.
    Example: when ScorableRanker scores 10 docs for query 'X', 
    you get 10 rows (one per doc).
    """

    __tablename__ = "scorable_ranks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # The query text (or hash) that initiated the ranking
    query_text: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # The scorable being ranked
    scorable_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    scorable_type: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # Total rank score
    rank_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Component scores (similarity, reward, recency, adaptability, etc.)
    components: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})

    # Embedding type used
    embedding_type: Mapped[str] = mapped_column(String, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query_text": self.query_text,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "rank_score": self.rank_score,
            "components": self.components,
            "embedding_type": self.embedding_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return (
            f"<ScorableRankORM(query='{self.query_text[:30]}...', "
            f"scorable={self.scorable_type}:{self.scorable_id}, "
            f"rank={self.rank_score:.3f})>"
        )
