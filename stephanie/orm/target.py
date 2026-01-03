# stephanie/orm/target.py
from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, Text, ForeignKey, Index, Float, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from stephanie.orm.base import Base


class TargetORM(Base):
    """
    Canonical output artifact (the “opposite” of a Source):
      - blog_post, paper_review, report, template, book_chapter, dataset, code_patch, etc.

    Key ideas:
      - pipeline_run_id ties back to pipeline_runs row (same int id you generate)
      - locator points to where the artifact lives (file path, URL, db key, etc.)
      - meta stores flexible payload (title, slug, tags, model name, prompt hash, etc.)
    """
    __tablename__ = "targets"

    id = Column(Integer, primary_key=True)

    # Tie to your pipeline_runs.id (recommended)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True, index=True)

    # “What did we produce?”
    target_type = Column(Text, nullable=False, index=True)      # blog_post | paper_review | book | template | report | ...
    target_format = Column(Text, nullable=True, index=True)     # markdown | html | pdf | json | docx | ...
    title = Column(Text, nullable=True)

    # Where it is stored / how to load it
    target_uri = Column(Text, nullable=False, index=True)       # file://... or relative path or https://...
    canonical_uri = Column(Text, nullable=True, index=True)

    # Lifecycle
    status = Column(Text, nullable=False, server_default="created", index=True)  # created|published|failed|archived
    content_hash = Column(Text, nullable=True, index=True)

    # Optional: attach to a graph root (if you want to anchor this target to a node)
    root_node_type = Column(Text, nullable=True, index=True)    # e.g. "paper", "paper_section", "concept_cluster"
    root_node_id = Column(Text, nullable=True, index=True)      # your scorable_id / node id

    meta = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_targets_run_type", "pipeline_run_id", "target_type"),
        Index("ix_targets_root", "root_node_type", "root_node_id"),
    )


class TargetInputORM(Base):
    """
    Many-to-many: targets -> sources

    Examples:
      relation_type = "derived_from" | "cites" | "uses" | "summarizes" | "evidence"
    """
    __tablename__ = "target_inputs"

    id = Column(Integer, primary_key=True)

    target_id = Column(Integer, ForeignKey("targets.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True)

    relation_type = Column(Text, nullable=False, index=True)
    weight = Column(Float, nullable=True)

    meta = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("target_id", "source_id", "relation_type", name="uq_target_inputs"),
        Index("ix_target_inputs_lookup", "target_id", "relation_type"),
    )
All right good