# stephanie/evaluation/adapters/stephanie_legacy.py
"""Read adapter over Stephanie's legacy evaluation stores (§23).

Precedence: normalized score rows first; JSON snapshot only when rows
are absent; divergence diagnostic when both exist and disagree.
Legacy rows have no ``is_active`` — treated as active during comparison.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from stephanie.evaluation.criterion import Criterion
from stephanie.evaluation.diagnostics import (
    LEGACY_SNAPSHOT_DIVERGENCE,
    EvaluationDiagnostic,
)
from stephanie.evaluation.evaluation import Evaluation, EvaluatorRef
from stephanie.evaluation.score import Score
from stephanie.evaluation.subject import SubjectRef

MATCH = "MATCH"
EXPECTED_DIVERGENCE = "EXPECTED_DIVERGENCE"
UNEXPECTED_DIVERGENCE = "UNEXPECTED_DIVERGENCE"
LEGACY_ONLY = "LEGACY_ONLY"
CANONICAL_ONLY = "CANONICAL_ONLY"


def scorable_to_subject(scorable: Any) -> SubjectRef:
    """Legacy Stephanie Scorable -> SubjectRef (§5)."""
    if isinstance(scorable, SubjectRef):
        return scorable
    if isinstance(scorable, dict):
        subject_type = scorable.get("target_type") or scorable.get("scorable_type") or "custom"
        subject_id = scorable.get("id") or scorable.get("scorable_id") or "unknown"
        text = scorable.get("text")
        meta = scorable.get("metadata") or scorable.get("meta") or {}
        return SubjectRef(subject_type=str(subject_type), subject_id=str(subject_id), text=text, metadata=dict(meta))
    target_type = getattr(scorable, "target_type", "custom")
    scorable_id = getattr(scorable, "id", "unknown")
    text = getattr(scorable, "text", None)
    meta = getattr(scorable, "meta", None) or getattr(scorable, "_metadata", None) or {}
    return SubjectRef(
        subject_type=str(target_type),
        subject_id=str(scorable_id),
        text=text,
        metadata=dict(meta) if isinstance(meta, dict) else {},
    )


def evaluation_orm_to_canonical(
    orm_row: Any, score_rows: Sequence[Any] | None = None
) -> tuple[Evaluation, list[Score], list[EvaluationDiagnostic]]:
    """Convert a legacy EvaluationORM row. Returns (evaluation, scores, diagnostics)."""
    diagnostics: list[EvaluationDiagnostic] = []
    scores_dict = getattr(orm_row, "scores", None) or {}
    subject = SubjectRef(
        subject_type=str(getattr(orm_row, "scorable_type", "custom")),
        subject_id=str(getattr(orm_row, "scorable_id", "unknown")),
    )
    evaluation = Evaluation(
        evaluation_id=f"legacy:{getattr(orm_row, 'id', 'unknown')}",
        subject=subject,
        criterion=Criterion(name=str(getattr(orm_row, "strategy", None) or "legacy")),
        evaluator=EvaluatorRef(name=str(getattr(orm_row, "evaluator_name", "ScoreEvaluator"))),
        created_at=getattr(orm_row, "created_at", None) or datetime.utcnow(),
        run_id=str(getattr(orm_row, "pipeline_run_id", None))
        if getattr(orm_row, "pipeline_run_id", None) is not None
        else None,
        model_id=str(getattr(orm_row, "model_name", None) or "") or None,
        metadata={"source": getattr(orm_row, "source", None), "agent_name": getattr(orm_row, "agent_name", None)},
        is_active=True,  # legacy rows have no is_active
    )
    scores: list[Score] = []
    if score_rows:
        for row in score_rows:
            scores.append(
                Score(
                    score_id=f"legacy-score:{getattr(row, 'id', 'unknown')}",
                    evaluation_id=evaluation.evaluation_id,
                    dimension=str(getattr(row, "dimension", "unknown")),
                    value=float(getattr(row, "score", 0.0) or 0.0),
                    weight=getattr(row, "weight", None),
                    source=getattr(row, "source", None),
                    rationale=getattr(row, "rationale", None),
                    metadata={},
                )
            )
        # Snapshot-vs-rows agreement check (§23 rule 3).
        snapshot_dims = {k: v for k, v in scores_dict.items() if not k.startswith("_")}
        row_dims = {s.dimension: s.value for s in scores}
        for dim, snap in snapshot_dims.items():
            snap_value = snap.get("score") if isinstance(snap, dict) else snap
            if dim in row_dims and isinstance(snap_value, (int, float)):
                if abs(float(snap_value) - row_dims[dim]) > 1e-9:
                    diagnostics.append(
                        EvaluationDiagnostic(
                            LEGACY_SNAPSHOT_DIVERGENCE,
                            f"snapshot[{dim}]={snap_value} != row value {row_dims[dim]}",
                            {"evaluation_id": evaluation.evaluation_id, "dimension": dim},
                        )
                    )
    else:
        # JSON-only legacy row: snapshot is the only signal (rule 2).
        for dim, snap in scores_dict.items():
            if dim.startswith("_"):
                continue
            value = snap.get("score") if isinstance(snap, dict) else snap
            if isinstance(value, (int, float)):
                scores.append(
                    Score(
                        score_id=f"legacy-snapshot:{evaluation.evaluation_id}:{dim}",
                        evaluation_id=evaluation.evaluation_id,
                        dimension=str(dim),
                        value=float(value),
                        source="legacy_snapshot",
                        metadata={"legacy_snapshot": True},
                    )
                )
    return evaluation, scores, diagnostics


class StephanieLegacyEvaluationReader:
    """Read adapter normalizing legacy store access (§23, Phase 2)."""

    def __init__(self, evaluation_store: Any, score_store: Any = None):
        self._evaluations = evaluation_store
        self._scores = score_store

    async def read_canonical(
        self, scorable: Any, criterion: str
    ) -> tuple[Evaluation | None, list[Score], list[EvaluationDiagnostic]]:
        subject = scorable_to_subject(scorable)
        latest = self._evaluations.get_latest_for_target(
            scorable_id=subject.subject_id, scorable_type=subject.subject_type
        )
        if latest is None:
            return None, [], []
        score_rows = None
        if self._scores is not None:
            try:
                score_rows = self._scores.get_scores_for_evaluation(latest.get("id"))
            except Exception:
                score_rows = None
        orm_like = _dict_to_orm_like(latest)
        return evaluation_orm_to_canonical(orm_like, score_rows)


def _dict_to_orm_like(data: dict) -> Any:
    class _Row:
        pass

    row = _Row()
    for key, value in data.items():
        setattr(row, key, value)
    return row
