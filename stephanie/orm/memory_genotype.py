# stephanie/orm/memory_genotype.py
from __future__ import annotations

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Index,
                        Integer, String, Text, func)
from sqlalchemy.orm import relationship

from stephanie.orm.base import Base


class MemoryGenotypeORM(Base):
    """
    A MemEvolve "genotype" = a memory policy/config specimen to be evaluated and evolved.

    Notes:
    - Keep `spec` fully flexible (JSON) so we can evolve any knobs without schema churn.
    - `is_active` is a convenience flag (you can keep only one active per "family"/domain if you want).
    """

    __tablename__ = "memory_genotypes"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Optional lineage
    parent_id = Column(Integer, ForeignKey("memory_genotypes.id", ondelete="SET NULL"), nullable=True, index=True)
    generation = Column(Integer, nullable=False, default=0)

    # Identity
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)

    # The actual genotype payload (all knobs)
    spec = Column(JSON, nullable=False, default=dict)

    # Convenience + light stats
    is_active = Column(Integer, nullable=False, default=0, index=True)
    fitness_mean = Column(Float, nullable=True)
    fitness_count = Column(Integer, nullable=False, default=0)

    tags = Column(JSON, nullable=False, default=list)

    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    parent = relationship("MemoryGenotypeORM", remote_side=[id], backref="children", lazy="selectin")
    evolution_runs = relationship(
        "MemoryEvolutionRunORM",
        back_populates="genotype",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "name": self.name,
            "description": self.description,
            "spec": self.spec or {},
            "is_active": int(self.is_active or 0),
            "fitness_mean": self.fitness_mean,
            "fitness_count": int(self.fitness_count or 0),
            "tags": self.tags or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MemoryEvolutionRunORM(Base):
    """
    One evaluation of a genotype (optionally tied to a pipeline run / suite / goal).
    """

    __tablename__ = "memory_evolution_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    genotype_id = Column(
        Integer,
        ForeignKey("memory_genotypes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Optional provenance / slicing
    run_id = Column(String, nullable=True, index=True)       # PipelineRunORM id or your run_id string
    goal_id = Column(String, nullable=True, index=True)      # GoalORM id if you use it
    suite = Column(String, nullable=False, default="batch_01")
    domain = Column(String, nullable=True)
    model_name = Column(String, nullable=True)

    # Metrics (raw / scalar)
    fitness = Column(Float, nullable=True)   # optional single aggregate if you compute one
    perf = Column(Float, nullable=True)
    cost = Column(Float, nullable=True)
    delay = Column(Float, nullable=True)
    epistemic = Column(Float, nullable=True)
    stability = Column(Float, nullable=True)
    memory_eff = Column(Float, nullable=True)

    # Deep provenance
    metrics = Column(JSON, nullable=False, default=dict)      # detailed metrics, histograms, etc.
    diagnosis = Column(JSON, nullable=False, default=dict)    # defect profile output
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    genotype = relationship("MemoryGenotypeORM", back_populates="evolution_runs", lazy="selectin")

    __table_args__ = (
        Index("ix_memory_evolution_runs_genotype_created", "genotype_id", "created_at"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "genotype_id": self.genotype_id,
            "run_id": self.run_id,
            "goal_id": self.goal_id,
            "suite": self.suite,
            "domain": self.domain,
            "model_name": self.model_name,
            "fitness": self.fitness,
            "metrics": {
                "perf": self.perf,
                "cost": self.cost,
                "delay": self.delay,
                "epistemic": self.epistemic,
                "stability": self.stability,
                "memory_eff": self.memory_eff,
                "extra": self.metrics or {},
            },
            "diagnosis": self.diagnosis or {},
            "notes": self.notes or "",
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
