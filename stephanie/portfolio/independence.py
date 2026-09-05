# stephanie/portfolio/independence.py
"""Failure correlation primitives (§18–§19, sequence 3.7).

Record shared failure events first; derive joint/conditional rates later.
P(B wrong | A wrong) matters more than raw accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from stephanie.portfolio.roles import PortfolioRole


@dataclass(frozen=True)
class FailureObservation:
    task_id: str

    model_id: str
    role: PortfolioRole

    criterion: str

    failed: bool

    failure_type: Optional[str] = None


def _by_model(observations: Sequence[FailureObservation], model_id: str) -> dict[str, bool]:
    return {o.task_id: o.failed for o in observations if o.model_id == model_id}


def joint_failure_rate(
    observations: Sequence[FailureObservation], model_a: str, model_b: str
) -> Optional[float]:
    a, b = _by_model(observations, model_a), _by_model(observations, model_b)
    shared = set(a) & set(b)
    if not shared:
        return None
    return sum(1 for t in shared if a[t] and b[t]) / len(shared)


def conditional_failure_rate(
    observations: Sequence[FailureObservation], model_a: str, model_b: str
) -> Optional[float]:
    """P(B wrong | A wrong). High values mean B adds little beyond A."""
    a, b = _by_model(observations, model_a), _by_model(observations, model_b)
    a_failed = {t for t in set(a) & set(b) if a[t]}
    if not a_failed:
        return None
    return sum(1 for t in a_failed if b[t]) / len(a_failed)


def failure_overlap(
    observations: Sequence[FailureObservation], model_a: str, model_b: str
) -> Optional[float]:
    """Jaccard overlap of failure sets. 1.0 = identical mistakes."""
    a_failed = {o.task_id for o in observations if o.model_id == model_a and o.failed}
    b_failed = {o.task_id for o in observations if o.model_id == model_b and o.failed}
    union = a_failed | b_failed
    if not union:
        return None
    return len(a_failed & b_failed) / len(union)


def unique_detection_rate(
    observations: Sequence[FailureObservation], model: str, others: Sequence[str]
) -> Optional[float]:
    """Share of tasks where model succeeded while ALL others failed."""
    mine = _by_model(observations, model)
    rest = [_by_model(observations, o) for o in others]
    eligible = [t for t in mine if all(t in r for r in rest)]
    if not eligible:
        return None
    return sum(1 for t in eligible if not mine[t] and all(r[t] for r in rest)) / len(eligible)
