from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column, Integer, String, DateTime, JSON, ForeignKey, LargeBinary,
    Float, Index, Text
)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base   # same as your chat models


# ============================================================
#  MetricGroupORM
#  One record per pipeline run
# ============================================================

class MetricGroupORM(Base):
    __tablename__ = "metric_groups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, unique=True)   # pipeline run ID
    created_at = Column(DateTime, default=datetime.now)
    meta = Column(JSON, default={})  # can contain metric_importance, core_metrics, etc.

    # relationships
    vectors = relationship("MetricVectorORM", back_populates="group", cascade="all, delete-orphan")
    deltas  = relationship("MetricDeltaORM",  back_populates="group", cascade="all, delete-orphan")
    vpms    = relationship("MetricVPMORM",    back_populates="group", cascade="all, delete-orphan")

    def to_dict(self, include_children=False):
        d = {
            "id": self.id,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "meta": self.meta or {},
        }
        if include_children:
            d["vectors"] = [v.to_dict() for v in self.vectors]
            d["deltas"]  = [d.to_dict() for d in self.deltas]
            d["vpms"]    = [v.to_dict(meta=False) for v in self.vpms]
        return d


# ============================================================
#  MetricVectorORM
#  Raw + reduced vectors for a Scorable in a run
# ============================================================

class MetricVectorORM(Base):
    __tablename__ = "metric_vectors"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(
        String, ForeignKey("metric_groups.run_id", ondelete="CASCADE"), nullable=False
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
        String, ForeignKey("metric_groups.run_id", ondelete="CASCADE"), nullable=False
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
        String, ForeignKey("metric_groups.run_id", ondelete="CASCADE"), nullable=False
    )
    scorable_id = Column(String, nullable=False)
    scorable_type = Column(String, nullable=False)
    dimension = Column(String, nullable=True)  # optional per-dimension VPM

    width  = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)

    # Actual VPM image bytes (PNG is recommended)
    image_bytes = Column(LargeBinary, nullable=False)

    # any metadata ZeroModel wants to store
    meta = Column(JSON, default={})

    created_at = Column(DateTime, default=datetime.now)

    group = relationship("MetricGroupORM", back_populates="vpms")

    __table_args__ = (
        Index("ix_metric_vpms_run_scorable_dim", "run_id", "scorable_id", "dimension"),
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
