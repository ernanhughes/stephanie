# stephanie/orm/experiment.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Index, Integer, String, UniqueConstraint)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stephanie.orm.base import Base  # your existing Base


class ExperimentORM(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(
        String(128), index=True
    )  # e.g., "verification_strategy"
    label: Mapped[Optional[str]] = mapped_column(
        String(128)
    )  # optional human label
    status: Mapped[str] = mapped_column(
        String(32), default="active"
    )  # active|paused|archived
    domain: Mapped[Optional[str]] = mapped_column(String(64))
    config: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # knobs (min_samples, p-value, etc.)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now
    )

    variants = relationship(
        "VariantORM", back_populates="experiment", cascade="all, delete-orphan"
    )
    model_snapshots = relationship(
        "ExperimentModelSnapshotORM",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )
    __table_args__ = (
        UniqueConstraint("name", "domain", name="uq_experiment_name_domain"),
    )


class VariantORM(Base):
    __tablename__ = "experiment_variants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(16))  # "A" | "B" | "C"…
    is_control: Mapped[bool] = mapped_column(Boolean, default=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # serialized Strategy, Arena settings, etc.
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now
    )

    experiment = relationship("ExperimentORM", back_populates="variants")
    trials = relationship(
        "TrialORM", back_populates="variant", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "experiment_id", "name", name="uq_variant_name_per_experiment"
        ),
    )


class TrialORM(Base):
    __tablename__ = "experiment_trials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    variant_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_variants.id", ondelete="CASCADE")
    )
    # linkage back to your world
    case_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    pipeline_run_id: Mapped[Optional[str]] = mapped_column(String(64))
    domain: Mapped[Optional[str]] = mapped_column(String(64))
    experiment_group = Column(
        String, nullable=True, index=True
    )  # e.g. "experimental", "control", "null"
    tags_used = Column(JSONB, default=list)  # e.g. ["paper:2106.09685"]

    assigned_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    performance: Mapped[Optional[float]] = mapped_column(
        Float
    )  # e.g., overall score
    tokens: Mapped[Optional[int]] = mapped_column(Integer)
    cost: Mapped[Optional[float]] = mapped_column(Float)
    wall_sec: Mapped[Optional[float]] = mapped_column(Float)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    variant = relationship("VariantORM", back_populates="trials")

    __table_args__ = (
        # ensures deterministic assignment per (experiment, case)
        UniqueConstraint(
            "variant_id", "case_id", name="uq_trial_variant_case"
        ),
        Index("ix_trials_completed", "completed_at"),
    )


class TrialMetricORM(Base):
    __tablename__ = "experiment_trial_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trial_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_trials.id", ondelete="CASCADE")
    )
    key: Mapped[str] = mapped_column(
        String(64)
    )  # e.g., "overall", "k", "g", "avg_gain"
    value: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now
    )

    __table_args__ = (Index("ix_trial_metric_key", "key"),)


class ExperimentModelSnapshotORM(Base):
    __tablename__ = "experiment_model_snapshots"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer, ForeignKey("experiments.id"), nullable=False, index=True
    )
    name = Column(
        String(128), nullable=False, index=True
    )  # e.g., "learning_strategy"
    domain = Column(
        String(128), nullable=True, index=True
    )  # optional scope key
    version = Column(Integer, nullable=False)  # you control increments
    payload = Column(JSON, nullable=False, default={})  # serialized knobs/data
    validation = Column(
        JSON, nullable=True
    )  # stats bundle that justified commit
    committed_at = Column(DateTime, nullable=False, default=datetime.now)
    created_at = Column(DateTime, nullable=False, default=datetime.now)

    experiment = relationship(
        "ExperimentORM", back_populates="model_snapshots"
    )

    __table_args__ = (
        Index(
            "ix_model_snapshots_unique",
            "experiment_id",
            "name",
            "domain",
            "version",
            unique=True,
        ),
    )
