# stephanie/orm/lightning.py
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

# If you already have a shared Base, import it instead of creating a new one:
# from stephanie.models.base import Base as BaseORM
BaseORM = declarative_base()


class LightningORM(BaseORM):
    __tablename__ = "agent_lightning"

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True, autoincrement=True)

    run_id: Mapped[str] = mapped_column(sa.Text, nullable=False, index=False)
    step_idx: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    kind: Mapped[str] = mapped_column(sa.Text, nullable=False)   # e.g., "heartbeat", "candidate", "score_update", "leaderboard", "checkpoint", "final"
    agent: Mapped[str] = mapped_column(sa.Text, nullable=False)  # class name of the emitting agent

    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"))

    created_at: Mapped[str] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )

    __table_args__ = (
        sa.Index("ix_lightning_run_step", "run_id", "step_idx"),
    )

    # -------- helpers --------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_idx": self.step_idx,
            "kind": self.kind,
            "agent": self.agent,
            "payload": self.payload or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<LightningORM id={self.id} run_id={self.run_id} step_idx={self.step_idx} "
            f"kind={self.kind} agent={self.agent}>"
        )

