# stephanie/evaluation/persistence/orm.py
"""Canonical persistence model (§19). Explicit v2 namespace — never claim
the legacy ``evaluations`` table (§27). String keys; no domain FKs."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class CanonicalBase(DeclarativeBase):
    pass


class EvaluationV2ORM(CanonicalBase):
    __tablename__ = "evaluation_v2"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    subject_type: Mapped[str] = mapped_column(Text, nullable=False)
    subject_id: Mapped[str] = mapped_column(Text, nullable=False)

    criterion_name: Mapped[str] = mapped_column(Text, nullable=False)
    criterion_version: Mapped[str | None] = mapped_column(Text, nullable=True)

    evaluator_name: Mapped[str] = mapped_column(Text, nullable=False)

    model_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    experiment_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_source: Mapped[str | None] = mapped_column(Text, nullable=True)

    interpretation_namespace: Mapped[str | None] = mapped_column(Text, nullable=True)
    interpretation_value: Mapped[str | None] = mapped_column(Text, nullable=True)

    supersedes_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_evalv2_subject", "subject_type", "subject_id"),
        Index("ix_evalv2_criterion", "criterion_name"),
        Index("ix_evalv2_model", "model_id"),
        Index("ix_evalv2_task", "task_type"),
        Index("ix_evalv2_run", "run_id"),
        Index("ix_evalv2_experiment", "experiment_id"),
        Index("ix_evalv2_created", "created_at"),
        Index("ix_evalv2_active", "is_active"),
    )

    scores: Mapped[list["EvaluationScoreV2ORM"]] = relationship(
        back_populates="evaluation", cascade="all, delete-orphan"
    )
    attributes: Mapped[list["EvaluationAttributeV2ORM"]] = relationship(
        back_populates="evaluation", cascade="all, delete-orphan"
    )


class EvaluationScoreV2ORM(CanonicalBase):
    __tablename__ = "evaluation_score_v2"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluation_v2.id", ondelete="CASCADE"), nullable=False
    )

    dimension: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    scale_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    scale_max: Mapped[float | None] = mapped_column(Float, nullable=True)

    weight: Mapped[float | None] = mapped_column(Float, nullable=True)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_source: Mapped[str | None] = mapped_column(Text, nullable=True)

    scorer: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)

    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    evaluation: Mapped[EvaluationV2ORM] = relationship(back_populates="scores")
    attributes: Mapped[list["ScoreAttributeV2ORM"]] = relationship(
        back_populates="score", cascade="all, delete-orphan"
    )


class EvaluationAttributeV2ORM(CanonicalBase):
    __tablename__ = "evaluation_attribute_v2"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    evaluation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluation_v2.id", ondelete="CASCADE"), nullable=False
    )

    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)

    source: Mapped[str | None] = mapped_column(Text, nullable=True)

    evaluation: Mapped[EvaluationV2ORM] = relationship(back_populates="attributes")


class ScoreAttributeV2ORM(CanonicalBase):
    __tablename__ = "score_attribute_v2"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    score_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluation_score_v2.id", ondelete="CASCADE"), nullable=False
    )

    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)

    score: Mapped[EvaluationScoreV2ORM] = relationship(back_populates="attributes")


class EvaluationEvidenceLinkV2ORM(CanonicalBase):
    __tablename__ = "evaluation_evidence_link_v2"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    evaluation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_id: Mapped[str] = mapped_column(String(128), nullable=False)
    relationship: Mapped[str] = mapped_column(Text, nullable=False, default="supports")


class FusionSpecV2ORM(CanonicalBase):
    __tablename__ = "fusion_spec_v2"

    fusion_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    method: Mapped[str] = mapped_column(Text, nullable=False)
    weights: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
