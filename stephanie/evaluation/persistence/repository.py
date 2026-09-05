# stephanie/evaluation/persistence/repository.py
"""SQLAlchemy-backed canonical repository (§19, §27)."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Sequence

from stephanie.evaluation.criterion import Criterion, ScoreScale
from stephanie.evaluation.evaluation import Evaluation, EvaluationAttribute, EvaluatorRef
from stephanie.evaluation.interpretation import Interpretation
from stephanie.evaluation.persistence.orm import (
    CanonicalBase,
    EvaluationAttributeV2ORM,
    EvaluationScoreV2ORM,
    EvaluationV2ORM,
    ScoreAttributeV2ORM,
)
from stephanie.evaluation.score import Score, ScoreAttribute
from stephanie.evaluation.subject import SubjectRef


def _encode_value(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


class SqlAlchemyEvaluationRepository:
    """Canonical store over the ``*_v2`` tables. Session factory injected."""

    def __init__(self, session_factory) -> None:
        self._sessions = session_factory

    # -- writes ---------------------------------------------------------

    async def append(self, evaluation: Evaluation, scores: Sequence[Score]) -> None:
        from sqlalchemy.exc import IntegrityError

        with self._sessions() as session:
            if session.get(EvaluationV2ORM, evaluation.evaluation_id) is not None:
                raise ValueError(f"duplicate evaluation_id: {evaluation.evaluation_id}")
            session.add(
                EvaluationV2ORM(
                    id=evaluation.evaluation_id,
                    subject_type=evaluation.subject.subject_type,
                    subject_id=evaluation.subject.subject_id,
                    criterion_name=evaluation.criterion.name,
                    criterion_version=evaluation.criterion.version,
                    evaluator_name=evaluation.evaluator.name,
                    model_id=evaluation.model_id,
                    task_type=evaluation.task_type,
                    run_id=evaluation.run_id,
                    experiment_id=evaluation.experiment_id,
                    confidence=evaluation.confidence,
                    confidence_source=evaluation.confidence_source,
                    interpretation_namespace=evaluation.interpretation.namespace
                    if evaluation.interpretation
                    else None,
                    interpretation_value=evaluation.interpretation.value
                    if evaluation.interpretation
                    else None,
                    supersedes_id=evaluation.supersedes_id,
                    is_active=evaluation.is_active,
                    meta=dict(evaluation.metadata or {}),
                    created_at=evaluation.created_at,
                )
            )
            for score in scores:
                session.add(
                    EvaluationScoreV2ORM(
                        id=score.score_id,
                        evaluation_id=evaluation.evaluation_id,
                        dimension=score.dimension,
                        value=score.value,
                        scale_min=score.scale.minimum if score.scale else None,
                        scale_max=score.scale.maximum if score.scale else None,
                        weight=score.weight,
                        confidence=score.confidence,
                        confidence_source=score.confidence_source,
                        scorer=score.scorer,
                        source=score.source,
                        rationale=score.rationale,
                        meta=dict(score.metadata or {}),
                        created_at=score.created_at or evaluation.created_at,
                    )
                )
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ValueError(f"duplicate key on append: {exc}") from exc

    async def deactivate(self, evaluation_id: str) -> None:
        with self._sessions() as session:
            row = session.get(EvaluationV2ORM, evaluation_id)
            if row is None:
                return
            row.is_active = False
            session.commit()

    async def add_evaluation_attributes(self, attrs: Sequence[EvaluationAttribute]) -> None:
        with self._sessions() as session:
            for attr in attrs:
                session.add(
                    EvaluationAttributeV2ORM(
                        evaluation_id=attr.evaluation_id,
                        namespace=attr.namespace,
                        name=attr.name,
                        value=_encode_value(attr.value),
                        source=attr.source,
                    )
                )
            session.commit()

    async def add_score_attributes(self, attrs: Sequence[ScoreAttribute]) -> None:
        with self._sessions() as session:
            for attr in attrs:
                session.add(
                    ScoreAttributeV2ORM(
                        score_id=attr.score_id,
                        namespace=attr.namespace,
                        name=attr.name,
                        value=_encode_value(attr.value),
                    )
                )
            session.commit()

    async def evaluation_attributes(
        self, evaluation_id: str
    ) -> Sequence[EvaluationAttribute]:
        with self._sessions() as session:
            rows = (
                session.query(EvaluationAttributeV2ORM)
                .filter(EvaluationAttributeV2ORM.evaluation_id == evaluation_id)
                .all()
            )
            return [
                EvaluationAttribute(
                    evaluation_id=row.evaluation_id,
                    namespace=row.namespace,
                    name=row.name,
                    value=row.value,
                    source=row.source,
                )
                for row in rows
            ]

    async def score_attributes(self, score_id: str) -> Sequence[ScoreAttribute]:
        with self._sessions() as session:
            rows = (
                session.query(ScoreAttributeV2ORM)
                .filter(ScoreAttributeV2ORM.score_id == score_id)
                .all()
            )
            return [
                ScoreAttribute(
                    score_id=row.score_id,
                    namespace=row.namespace,
                    name=row.name,
                    value=row.value,
                )
                for row in rows
            ]

    async def link_evidence(self, link) -> None:
        from stephanie.evaluation.persistence.orm import EvaluationEvidenceLinkV2ORM

        with self._sessions() as session:
            session.add(
                EvaluationEvidenceLinkV2ORM(
                    evaluation_id=link.evaluation_id,
                    evidence_id=link.evidence_id,
                    relationship=link.relationship,
                )
            )
            session.commit()

    async def performance_history(
        self,
        *,
        model_id: str | None = None,
        task_type: str | None = None,
        criterion: str | None = None,
        limit: int = 200,
    ) -> Sequence[Evaluation]:
        """Gate query (§33): how has this model performed on this kind of work?"""
        with self._sessions() as session:
            query = session.query(EvaluationV2ORM).filter(
                EvaluationV2ORM.is_active.is_(True)
            )
            if model_id:
                query = query.filter(EvaluationV2ORM.model_id == model_id)
            if task_type:
                query = query.filter(EvaluationV2ORM.task_type == task_type)
            if criterion:
                query = query.filter(EvaluationV2ORM.criterion_name == criterion)
            rows = query.order_by(EvaluationV2ORM.created_at.desc()).limit(limit).all()
            return [_to_evaluation(row) for row in rows]

    # -- reads ----------------------------------------------------------

    async def get(self, evaluation_id: str) -> Evaluation | None:
        with self._sessions() as session:
            row = session.get(EvaluationV2ORM, evaluation_id)
            return _to_evaluation(row) if row else None

    async def latest(self, subject: SubjectRef, criterion: str) -> Evaluation | None:
        with self._sessions() as session:
            rows = (
                session.query(EvaluationV2ORM)
                .filter(
                    EvaluationV2ORM.subject_type == subject.subject_type,
                    EvaluationV2ORM.subject_id == subject.subject_id,
                    EvaluationV2ORM.criterion_name == criterion,
                    EvaluationV2ORM.is_active.is_(True),
                )
                .order_by(EvaluationV2ORM.created_at.desc())
                .limit(1)
                .all()
            )
            return _to_evaluation(rows[0]) if rows else None

    async def list_for_subject(
        self, subject: SubjectRef, *, active_only: bool = True
    ) -> Sequence[Evaluation]:
        with self._sessions() as session:
            query = session.query(EvaluationV2ORM).filter(
                EvaluationV2ORM.subject_type == subject.subject_type,
                EvaluationV2ORM.subject_id == subject.subject_id,
            )
            if active_only:
                query = query.filter(EvaluationV2ORM.is_active.is_(True))
            return [_to_evaluation(row) for row in query.order_by(EvaluationV2ORM.created_at).all()]

    async def scores(self, evaluation_id: str) -> Sequence[Score]:
        with self._sessions() as session:
            rows = (
                session.query(EvaluationScoreV2ORM)
                .filter(EvaluationScoreV2ORM.evaluation_id == evaluation_id)
                .all()
            )
            return [_to_score(row) for row in rows]


def _to_evaluation(row: EvaluationV2ORM) -> Evaluation:
    interpretation = None
    if row.interpretation_value:
        interpretation = Interpretation(
            namespace=row.interpretation_namespace or "legacy",
            value=row.interpretation_value,
        )
    return Evaluation(
        evaluation_id=row.id,
        subject=SubjectRef(subject_type=row.subject_type, subject_id=row.subject_id),
        criterion=Criterion(name=row.criterion_name, version=row.criterion_version),
        evaluator=EvaluatorRef(name=row.evaluator_name),
        created_at=row.created_at,
        confidence=row.confidence,
        confidence_source=row.confidence_source,
        interpretation=interpretation,
        run_id=row.run_id,
        experiment_id=row.experiment_id,
        model_id=row.model_id,
        task_type=row.task_type,
        metadata=dict(row.meta or {}),
        is_active=row.is_active,
        supersedes_id=row.supersedes_id,
    )


def _to_score(row: EvaluationScoreV2ORM) -> Score:
    scale = None
    if row.scale_min is not None and row.scale_max is not None:
        scale = ScoreScale(minimum=row.scale_min, maximum=row.scale_max)
    return Score(
        score_id=row.id,
        evaluation_id=row.evaluation_id,
        dimension=row.dimension,
        value=row.value,
        scale=scale,
        weight=row.weight,
        confidence=row.confidence,
        confidence_source=row.confidence_source,
        scorer=row.scorer,
        source=row.source,
        rationale=row.rationale,
        created_at=row.created_at,
        metadata=dict(row.meta or {}),
    )
