# stephanie/data/strategy.py
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, cast

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class Strategy:
    """
    Immutable strategy configuration for verification and improvement.
    Domain object with behavior (validation, normalization, updates).
    """
    verification_threshold: float = 0.85
    skeptic_weight: float = 0.34
    editor_weight: float = 0.33
    risk_weight: float = 0.33
    version: int = 1

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not (0.7 <= self.verification_threshold <= 0.95):
            log.warning(
                "Verification threshold %.3f outside recommended range [0.70, 0.95]",
                self.verification_threshold
            )

        total_weight = self.skeptic_weight + self.editor_weight + self.risk_weight
        if abs(total_weight - 1.0) > 0.01:
            log.warning(
                "Strategy weights sum to %.3f (skeptic=%.3f, editor=%.3f, risk=%.3f); "
                "call .normalize() to renormalize.",
                total_weight, self.skeptic_weight, self.editor_weight, self.risk_weight
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Strategy:
        s = cls(
            verification_threshold=float(data.get("verification_threshold", 0.85)),
            skeptic_weight=float(data.get("skeptic_weight", 0.34)),
            editor_weight=float(data.get("editor_weight", 0.33)),
            risk_weight=float(data.get("risk_weight", 0.33)),
            version=int(data.get("version", 1)),
        )
        # Return normalized version to enforce invariants early
        return s.normalize()

    def to_dict(self) -> Dict[str, float]:
        return {
            "verification_threshold": self.verification_threshold,
            "skeptic_weight": self.skeptic_weight,
            "editor_weight": self.editor_weight,
            "risk_weight": self.risk_weight,
            "version": self.version,
        }

    def normalize(self) -> Strategy:
        sk = max(0.20, min(0.60, self.skeptic_weight))
        ed = max(0.20, min(0.60, self.editor_weight))
        rk = max(0.20, min(0.60, self.risk_weight))
        total = sk + ed + rk
        if total <= 0.01:
            return replace(self, skeptic_weight=0.34, editor_weight=0.33, risk_weight=0.33)
        sk, ed, rk = sk / total, ed / total, rk / total
        return cast(Strategy, replace(self, skeptic_weight=sk, editor_weight=ed, risk_weight=rk))

    def apply_changes(
        self,
        *,
        verification_threshold: Optional[float] = None,
        skeptic_weight: Optional[float] = None,
        editor_weight: Optional[float] = None,
        risk_weight: Optional[float] = None,
        version: Optional[int] = None,
        normalize: bool = True,
    ) -> Strategy:
        updated = replace(
            self,
            verification_threshold=verification_threshold if verification_threshold is not None else self.verification_threshold,
            skeptic_weight=skeptic_weight if skeptic_weight is not None else self.skeptic_weight,
            editor_weight=editor_weight if editor_weight is not None else self.editor_weight,
            risk_weight=risk_weight if risk_weight is not None else self.risk_weight,
            version=version if version is not None else self.version,
        )
        return updated.normalize() if normalize else updated

    def bump_version(self) -> Strategy:
        return cast(Strategy, replace(self, version=self.version + 1))

    def is_significantly_different(self, other: Strategy, threshold: float = 0.05) -> bool:
        return (
            abs(self.skeptic_weight - other.skeptic_weight) > threshold or
            abs(self.editor_weight - other.editor_weight) > threshold or
            abs(self.risk_weight - other.risk_weight) > threshold or
            abs(self.verification_threshold - other.verification_threshold) > threshold
        )
