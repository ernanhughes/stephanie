# stephanie/models/case_goal_state.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Index, Integer,
                        String, UniqueConstraint)

from stephanie.models.base import Base


class CaseGoalStateORM(Base):
    """
    Tracks the best (champion) result and A/B stats for a given (casebook, goal).

    - One row per (casebook_id, goal_id)
    - Stores the current champion case and its quality
    - Maintains an A/B run counter and simple trust metrics
    """
    __tablename__ = "case_goal_state"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Scope
    casebook_id = Column(Integer, ForeignKey("casebooks.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    goal_id     = Column(String,  nullable=False, index=True)

    # Champion
    champion_case_id  = Column(Integer, ForeignKey("cases.id", ondelete="SET NULL"),
                               nullable=True)
    champion_quality  = Column(Float, nullable=False, default=0.0)

    # A/B tracking & trust
    run_ix    = Column(Integer, nullable=False, default=0)     # how many runs recorded
    wins      = Column(Integer, nullable=False, default=0)     # CBR beat baseline
    losses    = Column(Integer, nullable=False, default=0)     # baseline beat CBR
    avg_delta = Column(Float,   nullable=False, default=0.0)   # EMA of (q_cbr - q_base)
    trust     = Column(Float,   nullable=False, default=0.0)   # [-1,1] convenience score

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now,
                        onupdate=datetime.now)

    __table_args__ = (
        UniqueConstraint("casebook_id", "goal_id", name="uq_case_goal_state"),
        Index("ix_case_goal", "casebook_id", "goal_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<CaseGoalStateORM(cb={self.casebook_id}, goal={self.goal_id}, "
            f"champion={self.champion_case_id}, q={self.champion_quality:.4f}, "
            f"run_ix={self.run_ix}, trust={self.trust:.3f})>"
        )

    # --- Convenience for A/B bookkeeping ---
    def update_ab_stats(self, improved: bool, delta: float, alpha: float = 0.2) -> None:
        """
        Update wins/losses and an exponential moving average of the quality delta.
        `delta` should be (q_cbr - q_baseline).
        """
        self.run_ix = (self.run_ix or 0) + 1
        if improved:
            self.wins = (self.wins or 0) + 1
        else:
            self.losses = (self.losses or 0) + 1

        prev = float(self.avg_delta or 0.0)
        self.avg_delta = (1.0 - alpha) * prev + alpha * float(delta)

        # Optional trust heuristic mapped to [-1, 1]
        v = self.avg_delta
        self.trust = max(-1.0, min(1.0, v))
