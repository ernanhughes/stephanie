# stephanie/evaluation/writer.py
"""Write-side helpers (§21). Append-only; corrections via supersede."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence
from uuid import uuid4

from stephanie.evaluation.evaluation import Evaluation, EvaluationObservation
from stephanie.evaluation.repository import EvaluationWriter
from stephanie.evaluation.score import Score


def build_evaluation(observation: EvaluationObservation, evaluation_id: str | None = None) -> Evaluation:
    return Evaluation(
        evaluation_id=evaluation_id or f"eval_{uuid4().hex[:12]}",
        subject=observation.subject,
        criterion=observation.criterion,
        evaluator=observation.evaluator,
        created_at=datetime.utcnow(),
        confidence=observation.confidence,
        confidence_source=observation.confidence_source,
        interpretation=observation.interpretation,
        run_id=observation.run_id,
        experiment_id=observation.experiment_id,
        model_id=observation.model_id,
        task_type=observation.task_type,
        metadata=dict(observation.metadata or {}),
    )


async def append_observation(
    writer: EvaluationWriter, observation: EvaluationObservation
) -> Evaluation:
    evaluation = build_evaluation(observation)
    await writer.append(evaluation, _bind_scores(evaluation, observation.scores))
    return evaluation


def _bind_scores(evaluation: Evaluation, scores: Sequence[Score]) -> list[Score]:
    from dataclasses import replace

    bound: list[Score] = []
    for score in scores:
        bound.append(
            replace(
                score,
                evaluation_id=evaluation.evaluation_id,
                score_id=score.score_id or f"score_{uuid4().hex[:12]}",
                created_at=score.created_at or evaluation.created_at,
            )
        )
    return bound


async def supersede(
    writer: EvaluationWriter, old: Evaluation, observation: EvaluationObservation
) -> Evaluation:
    """Record a correction: new row supersedes old; old deactivated."""
    from dataclasses import replace

    evaluation = replace(build_evaluation(observation), supersedes_id=old.evaluation_id)
    await writer.append(evaluation, _bind_scores(evaluation, observation.scores))
    await writer.deactivate(old.evaluation_id)
    return evaluation
