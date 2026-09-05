# stephanie/portfolio/outcome.py
"""Portfolio outcome (§16). Selection and synthesis stay different operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class PortfolioOutcome:
    outcome_id: str
    plan_id: str

    selected_candidate_id: Optional[str]

    synthesized_text: Optional[str]

    confidence: float | None

    candidate_ids: tuple[str, ...]

    disagreement_ids: tuple[str, ...]
    verification_ids: tuple[str, ...]

    total_cost_usd: Optional[Decimal]
    total_latency_ms: Optional[float]

    metadata: Mapping[str, Any] = field(default_factory=dict)
