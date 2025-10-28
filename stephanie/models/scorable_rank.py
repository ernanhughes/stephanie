# stephanie/models/scorable_rank.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stephanie.models.base import Base
from stephanie.models.evaluation import EvaluationORM


class ScorableRankORM(Base):
    """
    Stores ranking results for a single scorable under a query.
    Example: when ScorableRanker scores 10 docs for query 'X', 
    you get 10 rows (one per doc).
    """

    __tablename__ = "scorable_ranks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Query
    query_text: Mapped[str] = mapped_column(String, nullable=False)
    query_hash: Mapped[str] = mapped_column(String, index=True)  # optional dedup

    # Scorable reference
    scorable_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    scorable_type: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # Final rank score
    rank_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Raw + normalized component scores
    components: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})

    # Embedding backend used
    embedding_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Optional: link to EvaluationORM
    evaluation_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("evaluations.id"), nullable=True
    )
    evaluation: Mapped[Optional["EvaluationORM"]] = relationship(
        "EvaluationORM", foreign_keys=[evaluation_id]
    )

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    def to_dict(self, include_evaluation: bool = False) -> dict:
        data = {
            "id": self.id,
            "query_text": self.query_text,
            "query_hash": self.query_hash,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "rank_score": self.rank_score,
            "components": self.components,
            "embedding_type": self.embedding_type,
            "evaluation_id": self.evaluation_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_evaluation and self.evaluation:
            data["evaluation"] = self.evaluation.to_dict(include_relationships=False)
        return data

    def __repr__(self):
        return (
            f"<ScorableRankORM(query='{self.query_text[:30]}...', "
            f"scorable={self.scorable_type}:{self.scorable_id}, "
            f"rank={self.rank_score:.3f}, eval_id={self.evaluation_id})>"
        )
