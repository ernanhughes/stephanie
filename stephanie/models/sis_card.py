# stephanie/models/bus_event.py
from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import (JSON, Column, Float, Index, Integer, String,
                        UniqueConstraint)

from stephanie.models.base import Base


class SisCardORM(Base):
    __tablename__ = "sis_cards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(Float, nullable=False, index=True)  # epoch seconds
    scope = Column(String, nullable=False, index=True)  # e.g. "arena"
    key = Column(
        String, nullable=False, index=True
    )  # e.g. "paper:325|sec:abstract"
    title = Column(String, nullable=True)
    cards = Column(JSON, nullable=False)  # [{type,title,...}]
    meta = Column(
        JSON, nullable=True
    )  # case_id, paper_id, section_name, run_id, etc.
    hash = Column(String, nullable=True, unique=True)  # idempotency

    __table_args__ = (
        UniqueConstraint(
            "scope", "key", "hash", name="uq_sis_cards_scope_key_hash"
        ),
        Index("idx_sis_cards_scope_key_ts", "scope", "key", "ts"),
    )

    def to_dict(self, include_cards=True) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "ts": self.ts,
            "scope": self.scope,
            "key": self.key,
            "title": self.title,
            "meta": self.meta,
            "hash": self.hash,
        }
        if include_cards:
            d["cards"] = self.cards
        return d

