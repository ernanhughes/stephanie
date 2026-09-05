# stephanie/portfolio/history.py
"""Historical performance reporting (§17, sequence 3.6). Reporting only —
no automatic routing from these numbers until 3.8."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence

from stephanie.evaluation.evaluation import Evaluation
from stephanie.portfolio.roles import PortfolioRole


@dataclass(frozen=True)
class ModelRolePerformance:
    model_id: str
    task_type: str
    role: PortfolioRole

    criterion: str

    observations: int

    mean_score: Optional[float] = None
    mean_confidence: Optional[float] = None

    mean_cost: Optional[Decimal] = None
    mean_latency_ms: Optional[float] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)


async def role_performance(
    reader,
    *,
    task_type: str,
    criterion: str,
    role: PortfolioRole,
    model_id: Optional[str] = None,
) -> list[ModelRolePerformance]:
    """Aggregate model x task x role from the Stage 2.5 gate query."""
    history: Sequence[Evaluation] = await reader.performance_history(
        model_id=model_id, task_type=task_type, criterion=criterion
    )
    by_model: dict[str, list[Evaluation]] = {}
    for evaluation in history:
        if evaluation.metadata.get("portfolio_role") != role.value:
            continue
        by_model.setdefault(evaluation.model_id or "unknown", []).append(evaluation)
    report: list[ModelRolePerformance] = []
    for mid, evaluations in sorted(by_model.items()):
        confidences = [e.confidence for e in evaluations if e.confidence is not None]
        latencies = [
            e.metadata.get("latency_ms")
            for e in evaluations
            if isinstance(e.metadata.get("latency_ms"), (int, float))
        ]
        report.append(
            ModelRolePerformance(
                model_id=mid,
                task_type=task_type,
                role=role,
                criterion=criterion,
                observations=len(evaluations),
                mean_score=None,  # scores aggregated by caller via scores_by_evaluation
                mean_confidence=(sum(confidences) / len(confidences)) if confidences else None,
                mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
            )
        )
    return report
