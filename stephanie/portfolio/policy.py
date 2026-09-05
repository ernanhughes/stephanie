# stephanie/portfolio/policy.py
"""Portfolio policy: deterministic defaults now, history-driven later (§7)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from stephanie.portfolio.plan import PortfolioBudget


@dataclass(frozen=True)
class PortfolioPolicy:
    task_type: str

    preferred_primary: Optional[str] = None
    preferred_reviewers: tuple[str, ...] = ()

    require_different_provider_for_reviewer: bool = True
    include_breadth: bool = True
    include_independent_reviewer: bool = True

    budget: PortfolioBudget = field(default_factory=PortfolioBudget)

    synthesis_policy: str = "evidence_weighted_selection"

    # Later (§3.8): allow history to alter defaults. Off in 3.1.
    use_historical_performance: bool = False
