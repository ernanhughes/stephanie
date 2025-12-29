from __future__ import annotations

from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Index, Integer, LargeBinary, String, Text)
from sqlalchemy.orm import relationship

from stephanie.orm.base import Base  # same as your chat models


class MetricGroupORM(Base):
    __tablename__ = "metric_groups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, unique=True)  # pipeline run ID
    created_at = Column(DateTime, default=datetime.now)

    # Core Frontier Intelligence fields
    frontier_metric = Column(
        String, nullable=True
    )  # The selected frontier metric
    critic_status = Column(
        String, nullable=True
    )  # IMPROVING, STABLE, DEGRADING, STAGNANT
    critic_action = Column(
        String, nullable=True
    )  # WAIT, KEEP, TRAIN_MORE, PROMOTE, RESET
    critic_confidence = Column(
        Float, nullable=True
    )  # Confidence in the decision (0-1)
    is_best_model = Column(
        Boolean, default=False
    )  # Whether this run produced the best model
    model_version = Column(
        String, nullable=True
    )  # Version of the critic model

    # Decision metrics
    auc_score = Column(Float, nullable=True)  # AUC of the critic
    band_separation = Column(Float, nullable=True)  # Band separation score
    stability_score = Column(Float, nullable=True)  # Metric stability score
    feature_consistency = Column(
        Float, nullable=True
    )  # Feature consistency score

    # Additional metadata
    meta = Column(
        JSON, default={}
    )  # can contain metric_importance, core_metrics, etc.

    # relationships
    vectors = relationship(
        "MetricVectorORM", back_populates="group", cascade="all, delete-orphan"
    )
    deltas = relationship(
        "MetricDeltaORM", back_populates="group", cascade="all, delete-orphan"
    )
    vpms = relationship(
        "MetricVPMORM", back_populates="group", cascade="all, delete-orphan"
    )

    # New relationships for Frontier Intelligence
    critic_runs = relationship(
        "CriticRunORM", back_populates="group", cascade="all, delete-orphan"
    )
    critic_models = relationship(
        "CriticModelORM", back_populates="group", cascade="all, delete-orphan"
    )

    def to_dict(self, include_children=False):
        d = {
            "id": self.id,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            # Frontier Intelligence fields
            "frontier_metric": self.frontier_metric,
            "critic_status": self.critic_status,
            "critic_action": self.critic_action,
            "critic_confidence": self.critic_confidence,
            "is_best_model": self.is_best_model,
            "model_version": self.model_version,
            "auc_score": self.auc_score,
            "band_separation": self.band_separation,
            "stability_score": self.stability_score,
            "feature_consistency": self.feature_consistency,
            "meta": self.meta or {},
        }
        if include_children:
            d["vectors"] = [v.to_dict() for v in self.vectors]
            d["deltas"] = [d.to_dict() for d in self.deltas]
            d["vpms"] = [v.to_dict(meta=False) for v in self.vpms]
            d["critic_runs"] = [r.to_dict() for r in self.critic_runs]
            d["critic_models"] = [m.to_dict() for m in self.critic_models]
        return d


# ============================================================
#  MetricVectorORM
#  Raw + reduced vectors for a Scorable in a run
# ============================================================


class MetricVectorORM(Base):
    __tablename__ = "metric_vectors"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(
        String,
        ForeignKey("metric_groups.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    scorable_id = Column(String, nullable=False)
    scorable_type = Column(String, nullable=False)

    # raw metric values
    metrics = Column(JSON, default={})  # dict[str,float]

    # reduced or normalized representation
    reduced = Column(JSON, default={})  # dict[str,float]

    created_at = Column(DateTime, default=datetime.now)

    group = relationship("MetricGroupORM", back_populates="vectors")

    __table_args__ = (
        Index("ix_metric_vectors_run_scorable", "run_id", "scorable_id"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "metrics": self.metrics or {},
            "reduced": self.reduced or {},
            "created_at": self.created_at.isoformat(),
        }


# ============================================================
#  MetricDeltaORM
#  Target vs baseline for a scorable
# ============================================================


class MetricDeltaORM(Base):
    __tablename__ = "metric_deltas"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(
        String,
        ForeignKey("metric_groups.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    scorable_id = Column(String, nullable=False)
    scorable_type = Column(String, nullable=False)

    # diffs, ratios, etc.
    deltas = Column(JSON, default={})

    created_at = Column(DateTime, default=datetime.now)

    group = relationship("MetricGroupORM", back_populates="deltas")

    __table_args__ = (
        Index("ix_metric_deltas_run_scorable", "run_id", "scorable_id"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "deltas": self.deltas or {},
            "created_at": self.created_at.isoformat(),
        }


# ============================================================
#  MetricVPMORM
#  Stores visual policy maps produced by ZeroModel/VPM
# ============================================================


class MetricVPMORM(Base):
    __tablename__ = "metric_vpms"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(
        String,
        ForeignKey("metric_groups.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    scorable_id = Column(String, nullable=False)
    scorable_type = Column(String, nullable=False)
    dimension = Column(String, nullable=True)  # optional per-dimension VPM

    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)

    # Actual VPM image bytes (PNG is recommended)
    image_bytes = Column(LargeBinary, nullable=False)

    # any metadata ZeroModel wants to store
    meta = Column(JSON, default={})

    created_at = Column(DateTime, default=datetime.now)

    group = relationship("MetricGroupORM", back_populates="vpms")

    __table_args__ = (
        Index(
            "ix_metric_vpms_run_scorable_dim",
            "run_id",
            "scorable_id",
            "dimension",
        ),
    )

    def to_dict(self, meta=True):
        d = {
            "id": self.id,
            "run_id": self.run_id,
            "scorable_id": self.scorable_id,
            "scorable_type": self.scorable_type,
            "dimension": self.dimension,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at.isoformat(),
        }
        if meta:
            d["meta"] = self.meta or {}
        return d


class CriticRunORM(Base):
    """Tracks each critic run with detailed metrics and decisions"""

    __tablename__ = "critic_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to the metric group
    run_id = Column(
        String,
        ForeignKey("metric_groups.run_id", ondelete="CASCADE"),
        nullable=False,
    )

    # Run-specific data
    model_version = Column(String, nullable=False)
    auc = Column(Float, nullable=False)
    band_separation = Column(Float, nullable=True)
    stability_score = Column(Float, nullable=True)
    feature_consistency = Column(Float, nullable=True)
    is_promoted = Column(Boolean, default=False)

    # Decision data
    decision_action = Column(
        String, nullable=False
    )  # WAIT, KEEP, TRAIN_MORE, PROMOTE, RESET
    decision_confidence = Column(Float, nullable=True)
    decision_reason = Column(Text, nullable=True)
    decision_advice = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.now)

    # Relationship back to MetricGroupORM
    group = relationship("MetricGroupORM", back_populates="critic_runs")

    __table_args__ = (
        Index("ix_critic_runs_run_id", "run_id"),
        Index("ix_critic_runs_model_version", "model_version"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "model_version": self.model_version,
            "auc": self.auc,
            "band_separation": self.band_separation,
            "stability_score": self.stability_score,
            "feature_consistency": self.feature_consistency,
            "is_promoted": self.is_promoted,
            "decision_action": self.decision_action,
            "decision_confidence": self.decision_confidence,
            "decision_reason": self.decision_reason,
            "decision_advice": self.decision_advice,
            "created_at": self.created_at.isoformat(),
        }


class CriticModelORM(Base):
    """Tracks critic model versions and their performance metrics"""

    __tablename__ = "critic_models"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Model identity
    model_version = Column(String, unique=True, nullable=False)
    run_id = Column(
        String,
        ForeignKey("metric_groups.run_id", ondelete="CASCADE"),
        nullable=False,
    )

    # Performance metrics
    auc = Column(Float, nullable=False)
    band_separation = Column(Float, nullable=True)
    stability_score = Column(Float, nullable=True)
    feature_consistency = Column(Float, nullable=True)

    # Status flags
    is_active = Column(Boolean, default=False)
    is_best = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    promoted_at = Column(DateTime, nullable=True)

    # Relationship back to MetricGroupORM
    group = relationship("MetricGroupORM", back_populates="critic_models")

    __table_args__ = (
        Index("ix_critic_models_version", "model_version"),
        Index("ix_critic_models_status", "is_active", "is_best"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "auc": self.auc,
            "band_separation": self.band_separation,
            "stability_score": self.stability_score,
            "feature_consistency": self.feature_consistency,
            "is_active": self.is_active,
            "is_best": self.is_best,
            "created_at": self.created_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat()
            if self.promoted_at
            else None,
        }
