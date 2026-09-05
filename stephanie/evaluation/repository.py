# stephanie/evaluation/repository.py
"""Reader/Writer protocols + in-memory repository (§21).

Separate interfaces make the strangler easier: readers migrate first.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from stephanie.evaluation.evaluation import Evaluation, EvaluationAttribute
from stephanie.evaluation.evidence import EvaluationEvidenceLink, EvidenceRef
from stephanie.evaluation.score import Score, ScoreAttribute
from stephanie.evaluation.subject import SubjectRef


class EvaluationReader(Protocol):
    async def get(self, evaluation_id: str) -> Evaluation | None: ...
    async def latest(self, subject: SubjectRef, criterion: str) -> Evaluation | None: ...
    async def list_for_subject(
        self, subject: SubjectRef, *, active_only: bool = True
    ) -> Sequence[Evaluation]: ...
    async def scores(self, evaluation_id: str) -> Sequence[Score]: ...


class EvaluationWriter(Protocol):
    async def append(self, evaluation: Evaluation, scores: Sequence[Score]) -> None: ...
    async def deactivate(self, evaluation_id: str) -> None: ...


class InMemoryEvaluationRepository(EvaluationReader, EvaluationWriter):
    """Append/supersede lifecycle with cascade deletes (§18, §20)."""

    def __init__(self) -> None:
        self.evaluations: dict[str, Evaluation] = {}
        self._scores: dict[str, list[Score]] = {}
        self._eval_attrs: dict[str, list[EvaluationAttribute]] = {}
        self._score_attrs: dict[str, list[ScoreAttribute]] = {}
        self.evidence_links: list[EvaluationEvidenceLink] = []

    async def append(self, evaluation: Evaluation, scores: Sequence[Score]) -> None:
        if evaluation.evaluation_id in self.evaluations:
            raise ValueError(f"duplicate evaluation_id: {evaluation.evaluation_id} (append is idempotent-safe: retry with same id is rejected, correct via supersede)")
        self.evaluations[evaluation.evaluation_id] = evaluation
        self._scores[evaluation.evaluation_id] = list(scores)
        self._eval_attrs.setdefault(evaluation.evaluation_id, [])
        for score in scores:
            self._score_attrs.setdefault(score.score_id, [])

    async def deactivate(self, evaluation_id: str) -> None:
        from dataclasses import replace

        current = self.evaluations.get(evaluation_id)
        if current is None:
            return
        self.evaluations[evaluation_id] = replace(current, is_active=False)

    async def get(self, evaluation_id: str) -> Evaluation | None:
        return self.evaluations.get(evaluation_id)

    async def latest(self, subject: SubjectRef, criterion: str) -> Evaluation | None:
        candidates = [
            e
            for e in self.evaluations.values()
            if e.is_active
            and e.subject.key == subject.key
            and e.criterion.name == criterion
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.created_at)

    async def list_for_subject(
        self, subject: SubjectRef, *, active_only: bool = True
    ) -> Sequence[Evaluation]:
        return [
            e
            for e in self.evaluations.values()
            if e.subject.key == subject.key and (e.is_active or not active_only)
        ]

    async def scores(self, evaluation_id: str) -> Sequence[Score]:
        return list(self._scores.get(evaluation_id, []))

    async def add_evaluation_attributes(self, attrs: Sequence[EvaluationAttribute]) -> None:
        for attr in attrs:
            self._eval_attrs.setdefault(attr.evaluation_id, []).append(attr)

    async def add_score_attributes(self, attrs: Sequence[ScoreAttribute]) -> None:
        for attr in attrs:
            self._score_attrs.setdefault(attr.score_id, []).append(attr)

    async def link_evidence(self, link: EvaluationEvidenceLink) -> None:
        self.evidence_links.append(link)

    async def performance_history(
        self,
        *,
        model_id: str | None = None,
        task_type: str | None = None,
        criterion: str | None = None,
        limit: int = 200,
    ) -> Sequence[Evaluation]:
        """Gate query (§33): how has this model performed on this kind of work?"""
        matches = [
            e
            for e in self.evaluations.values()
            if e.is_active
            and (model_id is None or e.model_id == model_id)
            and (task_type is None or e.task_type == task_type)
            and (criterion is None or e.criterion.name == criterion)
        ]
        matches.sort(key=lambda e: e.created_at, reverse=True)
        return matches[:limit]

    async def purge(self, evaluation_id: str) -> None:
        """Hard delete with cascade (score attrs die with their evaluation)."""
        evaluation = self.evaluations.pop(evaluation_id, None)
        if evaluation is None:
            return
        scores = self._scores.pop(evaluation_id, [])
        for score in scores:
            self._score_attrs.pop(score.score_id, None)
        self._eval_attrs.pop(evaluation_id, None)
        self.evidence_links = [l for l in self.evidence_links if l.evaluation_id != evaluation_id]
