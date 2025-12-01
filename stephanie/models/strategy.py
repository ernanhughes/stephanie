# stephanie/models/strategy.py
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON as SA_JSON
from sqlalchemy import Column, DateTime, Float, Integer, String

from stephanie.models.base import Base

DEFAULT_PACS_WEIGHTS = {"skeptic": 0.34, "editor": 0.33, "risk": 0.33}

@dataclass
class StrategyProfile:
    verification_threshold: float = 0.90
    pacs_weights: Optional[Dict[str, float]] = None
    strategy_version: int = 1
    last_updated: float = 0.0

    def __post_init__(self):
        if self.pacs_weights is None:
            self.pacs_weights = dict(DEFAULT_PACS_WEIGHTS)
        if not self.last_updated:
            self.last_updated = time.time()

    def update(self, *, pacs_weights: Optional[Dict[str, float]] = None,
               verification_threshold: Optional[float] = None):
        if pacs_weights is not None:
            self.pacs_weights = pacs_weights
        if verification_threshold is not None:
            self.verification_threshold = verification_threshold
        self.strategy_version += 1
        self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyProfile":
        return cls(
            verification_threshold=float(d.get("verification_threshold", 0.90)),
            pacs_weights=dict(d.get("pacs_weights", DEFAULT_PACS_WEIGHTS)),
            strategy_version=int(d.get("strategy_version", 1)),
            last_updated=float(d.get("last_updated", time.time())),
        )

def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if isinstance(dt, datetime) else None


class StrategyProfileORM(Base):
    """
    A single active strategy profile per (agent_name, scope).
    Stores current weights/threshold + version. If you want history, add a second table later.
    """
    __tablename__ = "strategy_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)

    agent_name = Column(String, nullable=False, index=True)
    scope      = Column(String, nullable=False, index=True, default="default")

    # strategy payload
    pacs_weights = Column(SA_JSON, nullable=False)          # {"skeptic": 0.34, "editor": 0.33, "risk": 0.33}
    verification_threshold = Column(Float, nullable=False)  # e.g., 0.90
    strategy_version = Column(Integer, nullable=False, default=1)

    # bookkeeping
    created_at   = Column(DateTime, default=datetime.now, nullable=False)
    last_updated = Column(DateTime, default=datetime.now, nullable=False)
    meta = Column(SA_JSON, nullable=True)  # optional extra fields

    # ---- helpers ----
    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "scope": self.scope,
            "pacs_weights": dict(self.pacs_weights or {}),
            "verification_threshold": float(self.verification_threshold or 0.9),
            "strategy_version": int(self.strategy_version or 1),
            "created_at": _iso(self.created_at),
            "last_updated": _iso(self.last_updated),
            "meta": self.meta or {},
        }
