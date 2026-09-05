# stephanie/portfolio/candidate.py
"""Portfolio candidate (§4). Why each invocation exists, not just which model."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Optional

from stephanie.models.request import ModelRequest
from stephanie.portfolio.roles import PortfolioRole


@dataclass(frozen=True)
class PortfolioCandidate:
    candidate_id: str

    model_id: str
    role: PortfolioRole

    request: ModelRequest

    independence_group: Optional[str] = None

    estimated_cost: Optional[Decimal] = None
    estimated_latency_ms: Optional[float] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)
