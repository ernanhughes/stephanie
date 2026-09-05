# stephanie/evaluation/reader.py
"""Read-side convenience queries (§21, §30). No silent missing→0, ever."""
from __future__ import annotations

from typing import Optional, Sequence

from stephanie.evaluation.evaluation import Evaluation
from stephanie.evaluation.repository import EvaluationReader
from stephanie.evaluation.score import Score
from stephanie.evaluation.subject import SubjectRef


async def latest_score_for_dimension(
    reader: EvaluationReader,
    subject: SubjectRef,
    criterion: str,
    dimension: str,
) -> Optional[Score]:
    """Latest active evaluation's score for one dimension, or None if absent."""
    evaluation = await reader.latest(subject, criterion)
    if evaluation is None:
        return None
    for score in await reader.scores(evaluation.evaluation_id):
        if score.dimension == dimension:
            return score
    return None


async def history(
    reader: EvaluationReader, subject: SubjectRef, *, active_only: bool = True
) -> Sequence[Evaluation]:
    ordered = sorted(
        await reader.list_for_subject(subject, active_only=active_only),
        key=lambda e: e.created_at,
    )
    return ordered
