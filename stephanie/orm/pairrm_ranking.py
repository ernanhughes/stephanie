# stephanie/orm/pairrm_ranking.py
from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    Index,
    func,
)
from stephanie.orm.base import Base


class PairRMRankingORM(Base):
    """
    One row per candidate in a PairRM ranking result.

    Natural key:
      (scorable_type, scorable_id, tool_name, run_id, candidate_id)

    This matches how your tool persists:
      store.upsert(row) where row contains those keys + rank/score.
    """

    __tablename__ = "pairrm_rankings"

    id = Column(Integer, primary_key=True, autoincrement=True)

    scorable_type = Column(String, nullable=False, index=True)
    scorable_id = Column(Integer, nullable=False, index=True)

    tool_name = Column(String, nullable=False, index=True)
    run_id = Column(String, nullable=True, index=True)

    candidate_id = Column(String, nullable=False)
    candidate_index = Column(Integer, nullable=False, default=0)

    rank = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)

    meta = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint(
            "scorable_type",
            "scorable_id",
            "tool_name",
            "run_id",
            "candidate_id",
            name="uq_pairrm_rankings_nk",
        ),
        Index("ix_pairrm_rankings_scorable_tool", "scorable_type", "scorable_id", "tool_name"),
    )
