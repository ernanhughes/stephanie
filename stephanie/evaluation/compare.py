# stephanie/evaluation/compare.py
"""Dual-read comparison harness (§26 Phase 3).

Emits MATCH / EXPECTED_DIVERGENCE / UNEXPECTED_DIVERGENCE /
LEGACY_ONLY / CANONICAL_ONLY per (subject, criterion) query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from stephanie.evaluation.diagnostics import DUAL_READ_DIVERGENCE
from stephanie.evaluation.evaluation import Evaluation
from stephanie.evaluation.score import Score
from stephanie.evaluation.subject import SubjectRef

MATCH = "MATCH"
EXPECTED_DIVERGENCE = "EXPECTED_DIVERGENCE"
UNEXPECTED_DIVERGENCE = "UNEXPECTED_DIVERGENCE"
LEGACY_ONLY = "LEGACY_ONLY"
CANONICAL_ONLY = "CANONICAL_ONLY"


@dataclass
class ComparisonOutcome:
    verdict: str
    subject: SubjectRef
    criterion: str
    notes: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)


def compare_evaluations(
    subject: SubjectRef,
    criterion: str,
    legacy: tuple[Evaluation | None, Sequence[Score]] | None,
    canonical: tuple[Evaluation | None, Sequence[Score]] | None,
    *,
    expected_divergences: Sequence[str] = (),
) -> ComparisonOutcome:
    legacy_eval, legacy_scores = legacy if legacy else (None, [])
    canonical_eval, canonical_scores = canonical if canonical else (None, [])

    if legacy_eval is None and canonical_eval is None:
        return ComparisonOutcome(MATCH, subject, criterion, ["both absent"])
    if legacy_eval is None:
        return ComparisonOutcome(CANONICAL_ONLY, subject, criterion)
    if canonical_eval is None:
        return ComparisonOutcome(LEGACY_ONLY, subject, criterion)

    notes: list[str] = []
    legacy_dims = {s.dimension: s.value for s in legacy_scores}
    canonical_dims = {s.dimension: s.value for s in canonical_scores}
    if set(legacy_dims) != set(canonical_dims):
        notes.append(
            f"dimension set differs: legacy={sorted(legacy_dims)} canonical={sorted(canonical_dims)}"
        )
    for dim in set(legacy_dims) & set(canonical_dims):
        if abs(legacy_dims[dim] - canonical_dims[dim]) > 1e-9:
            notes.append(f"score[{dim}]: legacy={legacy_dims[dim]} canonical={canonical_dims[dim]}")
    if (legacy_eval.confidence is None) != (canonical_eval.confidence is None):
        notes.append("confidence presence differs (None stays missing, never coerced)")
    elif (
        legacy_eval.confidence is not None
        and abs(legacy_eval.confidence - (canonical_eval.confidence or 0.0)) > 1e-9
    ):
        notes.append("confidence value differs")
    if legacy_eval.is_active != canonical_eval.is_active:
        notes.append("active status differs")

    if not notes:
        return ComparisonOutcome(MATCH, subject, criterion)
    if all(any(exp in note for exp in expected_divergences) for note in notes) and expected_divergences:
        return ComparisonOutcome(EXPECTED_DIVERGENCE, subject, criterion, notes)
    return ComparisonOutcome(
        UNEXPECTED_DIVERGENCE,
        subject,
        criterion,
        notes,
        {"code": DUAL_READ_DIVERGENCE},
    )
