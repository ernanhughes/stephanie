# stephanie/evaluation/criterion.py
"""Criterion + score scale (§6–§7). Raw scores stay raw; no silent normalization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ScoreScale:
    minimum: float
    maximum: float

    higher_is_better: bool = True

    name: Optional[str] = None

    def contains(self, value: float) -> bool:
        lo, hi = (self.minimum, self.maximum) if self.minimum <= self.maximum else (self.maximum, self.minimum)
        return lo <= value <= hi


@dataclass(frozen=True)
class Criterion:
    name: str
    version: Optional[str] = None

    description: Optional[str] = None

    scale: Optional[ScoreScale] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}" if self.version else self.name
