# stephanie/services/daimon/risk_types.py
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Tuple


class RiskTier(IntEnum):
    """
    Canonical risk tiers used across Daimon guards and dashboards.

    Order is meaningful (LOW < MEDIUM < HIGH < CRITICAL) and used for
    comparisons and aggregations. Store the integer value in metrics.
    """
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

    @classmethod
    def from_name(cls, name: str) -> "RiskTier":
        return {
            "low": cls.LOW,
            "medium": cls.MEDIUM,
            "high": cls.HIGH,
            "critical": cls.CRITICAL,
        }[name.strip().lower()]

    @classmethod
    def from_score(cls, score: float,
                   medium: float = 0.20,
                   high: float = 0.40,
                   critical: float = 0.60) -> "RiskTier":
        """
        Map a normalized score (≈[0,1]) to a tier using defaults shared across guards.
        """
        if score >= critical:
            return cls.CRITICAL
        if score >= high:
            return cls.HIGH
        if score >= medium:
            return cls.MEDIUM
        return cls.LOW

    def label(self) -> str:
        return {
            RiskTier.LOW: "LOW",
            RiskTier.MEDIUM: "MEDIUM",
            RiskTier.HIGH: "HIGH",
            RiskTier.CRITICAL: "CRITICAL",
        }[self]

    def color(self) -> str:
        """
        UI helper (hex). Keep consistent with SIS/GAP theming.
        """
        return {
            RiskTier.LOW: "#4CAF50",       # green
            RiskTier.MEDIUM: "#FFC107",    # amber
            RiskTier.HIGH: "#FF7043",      # deep orange
            RiskTier.CRITICAL: "#D32F2F",  # red
        }[self]


@dataclass(frozen=True)
class RiskAssessment:
    """
    Standard envelope for guard outputs.
    """
    tier: RiskTier
    score: float
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "tier": int(self.tier),
            "tier_label": self.tier.label(),
            "score": float(self.score),
            "reasons": list(self.reasons),
        }


# ----------------------------- Aggregation utils -----------------------------

def max_tier(tiers: Iterable[RiskTier]) -> RiskTier:
    """
    Conservative aggregation: overall risk is the maximum tier observed.
    """
    max_val = RiskTier.LOW
    for t in tiers:
        if t > max_val:
            max_val = t
    return max_val


def min_tier(tiers: Iterable[RiskTier]) -> RiskTier:
    """
    Optimistic aggregation: overall risk is the minimum tier observed.
    """
    min_val = RiskTier.CRITICAL
    empty = True
    for t in tiers:
        empty = False
        if t < min_val:
            min_val = t
    return RiskTier.LOW if empty else min_val


def combine_scores_weighted(pairs: Iterable[tuple[float, float]]) -> float:
    """
    Weighted mean of scores [(score, weight), ...]; defaults to 0 if empty.
    """
    num = 0.0
    den = 0.0
    for score, w in pairs:
        num += float(score) * float(w)
        den += float(w)
    return num / den if den > 0 else 0.0
