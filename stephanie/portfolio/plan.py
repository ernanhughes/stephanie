# stephanie/portfolio/plan.py
"""Portfolio plan + budget (§5–§6). Inspectable and replayable."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Optional

from stephanie.portfolio.candidate import PortfolioCandidate


@dataclass(frozen=True)
class PortfolioBudget:
    max_models: Optional[int] = None
    max_cost_usd: Optional[Decimal] = None
    max_tokens: Optional[int] = None
    max_latency_ms: Optional[float] = None


@dataclass(frozen=True)
class PortfolioPlan:
    plan_id: str

    task_type: str

    candidates: tuple[PortfolioCandidate, ...]

    budget: PortfolioBudget

    synthesis_policy: str

    created_at: datetime

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def rationale(self) -> list[str]:
        """Why each candidate exists — the explainability spine (§26.12)."""
        return [
            f"{c.candidate_id}: {c.role.value} <- {c.model_id}"
            + (f" [{c.independence_group}]" if c.independence_group else "")
            + (f" ({c.metadata['reason']})" if "reason" in c.metadata else "")
            for c in self.candidates
        ]
