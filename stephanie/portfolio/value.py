# stephanie/portfolio/value.py
"""Marginal intelligence value (§20–§21). Record components first;
no single opaque scalar until the components are trusted."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class UniqueContribution:
    contribution_id: str
    model_id: str
    given_portfolio: tuple[str, ...]

    kind: str  # new_issue | challenged_assumption | missing_evidence |
               # proposed_test | correct_alternative | rejected_primary

    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarginalValueComponents:
    model_id: str

    expected_quality_gain: Optional[float] = None
    expected_verification_gain: Optional[float] = None
    expected_unique_detection: Optional[float] = None

    expected_cost_usd: Optional[Decimal] = None
    expected_latency_ms: Optional[float] = None
    correlated_failure_penalty: Optional[float] = None

    def marginal_intelligence_value(
        self,
        *,
        quality_weight: float = 1.0,
        verification_weight: float = 1.5,
        detection_weight: float = 1.0,
        cost_weight: float = 1.0,
        latency_weight_per_s: float = 0.01,
        correlation_weight: float = 1.0,
    ) -> Optional[float]:
        """Transparent MIV: all weights explicit, all components inspectable."""
        gains = [
            (self.expected_quality_gain or 0.0) * quality_weight,
            (self.expected_verification_gain or 0.0) * verification_weight,
            (self.expected_unique_detection or 0.0) * detection_weight,
        ]
        penalties = [
            float(self.expected_cost_usd or 0) * cost_weight * 100.0,
            ((self.expected_latency_ms or 0.0) / 1000.0) * latency_weight_per_s,
            (self.correlated_failure_penalty or 0.0) * correlation_weight,
        ]
        if all(v is None for v in (self.expected_quality_gain,
                                    self.expected_verification_gain,
                                    self.expected_unique_detection)):
            return None
        return sum(gains) - sum(penalties)
