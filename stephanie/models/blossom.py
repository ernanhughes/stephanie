# stephanie/models/blossom.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Index, Integer, String, Text)
from sqlalchemy.orm import relationship

from stephanie.models.base import Base


class BlossomORM(Base):
    """
    A Blossom run: exploding a seed thought/trace into a graph (GoT/ToT),
    sharpening candidates, scoring, and selecting improved paths.
    """
    __tablename__ = "blossoms"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Linkage
    goal_id = Column(Integer, ForeignKey("goals.id", ondelete="SET NULL"), nullable=True)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True)

    # Provenance
    agent_name = Column(String, nullable=False)          # e.g., "NexusInlineAgent"
    strategy = Column(String, default="got")             # "got" | "tot" | "mixed"
    seed_type = Column(String, nullable=True)            # e.g., "scorable", "nexus_node", "document_section"
    seed_id = Column(String, nullable=True)              # free-form id (UUID/int/str)

    # Lifecycle
    status = Column(String, default="pending")           # pending|running|completed|failed|aborted
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    # Config snapshot + run stats
    params = Column(JSON, nullable=True)                 # cfg snapshot for reproducibility
    stats = Column(JSON, nullable=True)                  # counts, rates, etc.
    extra_data = Column(JSON, nullable=True)

    # Optional pointer to the root node of this blossom
    root_node_id = Column(Integer, ForeignKey("blossom_nodes.id", ondelete="SET NULL"), nullable=True)

    # --- Relationships (explicit to avoid ambiguity) ---
    nodes = relationship(
        "BlossomNodeORM",
        back_populates="blossom",
        cascade="all, delete-orphan",
        passive_deletes=True,
        primaryjoin="BlossomORM.id == BlossomNodeORM.blossom_id",
        foreign_keys="[BlossomNodeORM.blossom_id]",
    )

    edges = relationship(
        "BlossomEdgeORM",
        back_populates="blossom",
        cascade="all, delete-orphan",
        passive_deletes=True,
        primaryjoin="BlossomORM.id == BlossomEdgeORM.blossom_id",
        foreign_keys="[BlossomEdgeORM.blossom_id]",
    )

    root_node = relationship(
        "BlossomNodeORM",
        uselist=False,
        primaryjoin="BlossomNodeORM.id == BlossomORM.root_node_id",
        foreign_keys="[BlossomORM.root_node_id]",
        post_update=True,  # helps with circular FKs when root changes
    )

    def to_dict(self, include_children: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "goal_id": self.goal_id,
            "pipeline_run_id": self.pipeline_run_id,
            "agent_name": self.agent_name,
            "strategy": self.strategy,
            "seed_type": self.seed_type,
            "seed_id": self.seed_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "params": self.params,
            "stats": self.stats,
            "extra_data": self.extra_data,
            "root_node_id": self.root_node_id,
        }
        if include_children:
            data["nodes"] = [n.to_dict() for n in self.nodes]
            data["edges"] = [e.to_dict() for e in self.edges]
        return data

    def __repr__(self):
        return f"<Blossom id={self.id} status={self.status} seed={self.seed_type}:{self.seed_id}>"


class BlossomNodeORM(Base):
    """
    A thought node inside a Blossom graph.
    """
    __tablename__ = "blossom_nodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    blossom_id = Column(Integer, ForeignKey("blossoms.id", ondelete="CASCADE"), nullable=False)

    # Tree/graph structure
    parent_id = Column(Integer, ForeignKey("blossom_nodes.id", ondelete="SET NULL"), nullable=True)
    depth = Column(Integer, default=0)
    order_index = Column(Integer, default=0)  # sibling order for deterministic traversal

    # Prompt provenance (ties into Prompt/PromptProgram tables)
    prompt_id = Column(Integer, ForeignKey("prompts.id", ondelete="SET NULL"), nullable=True)
    prompt_program_id = Column(String, ForeignKey("prompt_programs.id", ondelete="SET NULL"), nullable=True)

    # Content
    state_text = Column(Text, nullable=True)
    sharpened_text = Column(Text, nullable=True)
    accepted = Column(Boolean, default=True)
    rationale = Column(Text, nullable=True)

    # Scores & features
    scores = Column(JSON, nullable=True)       # {"mrq": x, "sicql": y, "hrm": z, ...}
    features = Column(JSON, nullable=True)     # 2048-dim or mixed metrics
    tags = Column(JSON, nullable=True)         # ["sharpened","merged","refined",...]

    # Sharpening audit
    sharpen_passes = Column(Integer, default=0)
    sharpen_gain = Column(Float, nullable=True)
    sharpen_meta = Column(JSON, nullable=True)

    extra_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # --- Relationships (explicit to avoid ambiguity) ---
    blossom = relationship(
        "BlossomORM",
        back_populates="nodes",
        primaryjoin="BlossomORM.id == BlossomNodeORM.blossom_id",
        foreign_keys="[BlossomNodeORM.blossom_id]",
    )

    parent = relationship("BlossomNodeORM", remote_side=[id])

    edges_out = relationship(
        "BlossomEdgeORM",
        foreign_keys="BlossomEdgeORM.src_node_id",
        back_populates="src_node",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    edges_in = relationship(
        "BlossomEdgeORM",
        foreign_keys="BlossomEdgeORM.dst_node_id",
        back_populates="dst_node",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "blossom_id": self.blossom_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "order_index": self.order_index,
            "prompt_id": self.prompt_id,
            "prompt_program_id": self.prompt_program_id,
            "state_text": self.state_text,
            "sharpened_text": self.sharpened_text,
            "accepted": self.accepted,
            "rationale": self.rationale,
            "scores": self.scores,
            "features": self.features,
            "tags": self.tags,
            "sharpen_passes": self.sharpen_passes,
            "sharpen_gain": self.sharpen_gain,
            "sharpen_meta": self.sharpen_meta,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"<BlossomNode id={self.id} blossom={self.blossom_id} depth={self.depth}>"


Index("ix_blossom_nodes_blossom_depth", BlossomNodeORM.blossom_id, BlossomNodeORM.depth)

# Children lookups by (blossom_id, parent_id, order_index)
Index(
    "ix_blossom_nodes_children",
    BlossomNodeORM.blossom_id,
    BlossomNodeORM.parent_id,
    BlossomNodeORM.order_index,
    BlossomNodeORM.id,
)


class BlossomEdgeORM(Base):
    """
    A labeled edge between two Blossom nodes.
    """
    __tablename__ = "blossom_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    blossom_id = Column(Integer, ForeignKey("blossoms.id", ondelete="CASCADE"), nullable=False)

    src_node_id = Column(Integer, ForeignKey("blossom_nodes.id", ondelete="CASCADE"), nullable=False)
    dst_node_id = Column(Integer, ForeignKey("blossom_nodes.id", ondelete="CASCADE"), nullable=False)

    relation = Column(String, nullable=False, default="expand")   # expand|refine|merge|loop|prune|select
    score = Column(Float, nullable=True)                          # optional edge weight
    rationale = Column(Text, nullable=True)
    extra_data = Column(JSON, nullable=True)

    # --- Relationships (explicit to avoid ambiguity) ---
    blossom = relationship(
        "BlossomORM",
        back_populates="edges",
        primaryjoin="BlossomORM.id == BlossomEdgeORM.blossom_id",
        foreign_keys="[BlossomEdgeORM.blossom_id]",
    )
    src_node = relationship(
        "BlossomNodeORM",
        foreign_keys="[BlossomEdgeORM.src_node_id]",
        back_populates="edges_out",
    )
    dst_node = relationship(
        "BlossomNodeORM",
        foreign_keys="[BlossomEdgeORM.dst_node_id]",
        back_populates="edges_in",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "blossom_id": self.blossom_id,
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "relation": self.relation,
            "score": self.score,
            "rationale": self.rationale,
            "extra_data": self.extra_data,
        }

    def __repr__(self):
        return f"<BlossomEdge {self.src_node_id} -> {self.dst_node_id} ({self.relation})>"


# Fast edge queries + optional uniqueness per (blossom,src,dst)
Index(
    "ix_blossom_edges_triplet",
    BlossomEdgeORM.blossom_id,
    BlossomEdgeORM.src_node_id,
    BlossomEdgeORM.dst_node_id,
)

# Relation filter within a blossom
Index(
    "ix_blossom_edges_relation",
    BlossomEdgeORM.blossom_id,
    BlossomEdgeORM.relation,
)
