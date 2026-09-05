# stephanie/evaluation/adapters/writer_legacy.py
"""Read adapter over Writer's evaluation DTOs/stores (§24).

Writer ``verdict`` maps to evaluation-level Interpretation ONLY when it is
an evaluation verdict. Experiment KEEP/REVERT decisions are quarantined —
they belong to the experiment runtime, not to Evaluation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from stephanie.evaluation.criterion import Criterion
from stephanie.evaluation.diagnostics import VERDICT_AMBIGUOUS, EvaluationDiagnostic
from stephanie.evaluation.evaluation import Evaluation, EvaluatorRef
from stephanie.evaluation.interpretation import Interpretation
from stephanie.evaluation.score import Score
from stephanie.evaluation.subject import SubjectRef

# Experiment-decision vocabulary that must NOT become Interpretation.
EXPERIMENT_DECISIONS = frozenset({"KEEP", "REVERT", "BRANCH", "ACCEPT", "REJECT", "ESCALATE", "REVIEW"})


def writer_scorable_to_subject(scorable: Any) -> SubjectRef:
    if isinstance(scorable, SubjectRef):
        return scorable
    if isinstance(scorable, dict):
        return SubjectRef(
            subject_type=str(scorable.get("scorable_type", "text")),
            subject_id=str(scorable.get("scorable_id", "unknown")),
            text=scorable.get("text") or scorable.get("scorable_text"),
            metadata=dict(scorable.get("meta") or scorable.get("scorable_meta") or {}),
        )
    return SubjectRef(
        subject_type=str(getattr(scorable, "scorable_type", "text")),
        subject_id=str(getattr(scorable, "scorable_id", "unknown")),
        text=getattr(scorable, "scorable_text", None),
        metadata=dict(getattr(scorable, "scorable_meta", None) or {}),
    )


def dimension_dto_to_score(dto: Any, evaluation_id: str) -> Score:
    values = dto if isinstance(dto, dict) else _dto_to_dict(dto)
    return Score(
        score_id=f"writer:{evaluation_id}:{values.get('dimension', 'unknown')}",
        evaluation_id=evaluation_id,
        dimension=str(values.get("dimension", "unknown")),
        value=float(values.get("score", 0.0) or 0.0),
        confidence=values.get("confidence"),
        confidence_source="writer_dto",
        scorer=values.get("scorer"),
        rationale=values.get("rationale"),
        metadata={"raw": values.get("raw", {})},
    )


def verdict_to_interpretation(
    verdict: Any, *, namespace: str = "writer"
) -> tuple[Interpretation | None, EvaluationDiagnostic | None]:
    """Map Writer verdict strings; quarantine experiment decisions (§24)."""
    if not verdict:
        return None, None
    text = str(verdict)
    if text in EXPERIMENT_DECISIONS:
        return None, EvaluationDiagnostic(
            VERDICT_AMBIGUOUS,
            f"verdict {text!r} looks like an experiment decision, not an evaluation interpretation",
            {"verdict": text},
        )
    return Interpretation(namespace=namespace, value=text), None


def score_result_dto_to_canonical(
    dto: Any, subject: SubjectRef, evaluation_id: str
) -> tuple[Evaluation, list[Score], list[EvaluationDiagnostic]]:
    values = dto if isinstance(dto, dict) else _dto_to_dict(dto)
    diagnostics: list[EvaluationDiagnostic] = []
    interpretation, diag = verdict_to_interpretation(values.get("verdict"))
    if diag is not None:
        diagnostics.append(diag)
    evaluation = Evaluation(
        evaluation_id=evaluation_id,
        subject=subject,
        criterion=Criterion(name=str(values.get("goal", "legacy"))),
        evaluator=EvaluatorRef(name="writer_legacy"),
        created_at=datetime.utcnow(),
        confidence=values.get("confidence"),
        confidence_source="writer_dto" if values.get("confidence") is not None else None,
        interpretation=interpretation,
        model_id=values.get("target_id"),
        metadata=dict(values.get("meta", {}) or {}),
    )
    scores = [dimension_dto_to_score(d, evaluation_id) for d in values.get("scores", [])]
    return evaluation, scores, diagnostics


def _dto_to_dict(dto: Any) -> dict:
    if hasattr(dto, "model_dump"):
        return dto.model_dump()
    if hasattr(dto, "__dict__"):
        return dict(dto.__dict__)
    return {}
