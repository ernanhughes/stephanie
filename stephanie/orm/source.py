from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from stephanie.orm.base import Base


class SourceORM(Base):
    """
    Canonical provenance source.

    Examples:
      - source_type='url', locator='https://arxiv.org/pdf/2506.21734.pdf'
      - source_type='file', locator='C:\\papers\\2506.21734.pdf'
      - source_type='db', locator='postgres://.../table=paper_sections'
      - source_type='ai', locator='ollama/qwen3' (meta includes prompt_hash, params, etc.)
      - source_type='generated', locator='runs/paper_blogs/9791/.../section_packs.json'
    """
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True)

    # url | file | db | ai | generated | user | derived (you decide the vocabulary)
    source_type = Column(String(64), nullable=False, index=True)

    # The “where” string: URL, file path, db uri, model name, artifact path, etc.
    locator = Column(Text, nullable=False)

    # Optional normalization / dedupe helpers
    canonical_locator = Column(Text, nullable=True, index=True)
    content_hash = Column(String(128), nullable=True, index=True)   # sha256, etc.
    mime_type = Column(String(128), nullable=True)

    # Optional human fields
    name = Column(String(256), nullable=True)
    description = Column(Text, nullable=True)

    # Everything else goes here (tool, query, model params, run_id, etc.)
    meta = Column(JSONB, nullable=False, server_default="{}")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Backref links
    links = relationship("ScorableSourceLinkORM", back_populates="source", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("source_type", "locator", name="uq_sources_type_locator"),
        Index("ix_sources_type_canonical", "source_type", "canonical_locator"),
    )


class ScorableSourceLinkORM(Base):
    """
    Polymorphic link: attach sources to ANY scorable node.

    role examples:
      - origin        (this node IS this source)
      - fetched_from  (retrieved from)
      - derived_from  (transformed from)
      - generated_by  (created by AI/tool)
      - scored_by     (score produced by)
      - evidence      (edge/claim supported by)
    """
    __tablename__ = "scorable_sources"

    id = Column(Integer, primary_key=True)

    scorable_type = Column(String(64), nullable=False, index=True)   # e.g. "paper", "section", "edge"
    scorable_id = Column(Text, nullable=False, index=True)           # keep TEXT to avoid truncation

    source_id = Column(Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(64), nullable=False, server_default="origin", index=True)

    # optional: how strongly this source supports the link
    confidence = Column(Float, nullable=True)

    # optional: tie back to run that created this link
    pipeline_run_id = Column(Integer, nullable=True, index=True)

    meta = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    source = relationship("SourceORM", back_populates="links")

    __table_args__ = (
        UniqueConstraint("scorable_type", "scorable_id", "source_id", "role", name="uq_scorable_sources_link"),
        Index("ix_scorable_sources_lookup", "scorable_type", "scorable_id", "role"),
    )


class SourceCandidateORM(Base):
    """
    A specific search episode result (candidate) for a given run + goal + query.

    This is *not* the canonical Source. This is “the world showed me this URL
    for this query at this time.”

    Depends on:
      - sources(id)
    """
    __tablename__ = "source_candidates"

    id = Column(Integer, primary_key=True)

    pipeline_run_id = Column(Integer, nullable=False, index=True)

    goal_type = Column(String(64), nullable=False, index=True)
    query_text = Column(Text, nullable=False)

    # stable fingerprint for dedupe within a run
    query_hash = Column(String(64), nullable=False, index=True)

    source_id = Column(Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True)

    # rank as returned by the search provider (0-based or 1-based; you choose)
    rank = Column(Integer, nullable=True)

    title = Column(Text, nullable=True)
    snippet = Column(Text, nullable=True)

    # paper_pdf | paper_html | documentation | blog | wiki | code_repo | pdf | unknown
    result_type = Column(String(64), nullable=False, server_default="unknown", index=True)

    # google | brave | serpapi | arxiv | internal | ...
    provider = Column(String(64), nullable=True, index=True)

    # pending | selected | fetched | parsed | skipped | failed
    status = Column(String(32), nullable=False, server_default="pending", index=True)

    # convenience: the total quality score at time of scoring (0..1)
    quality_total = Column(Float, nullable=True)

    meta = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "pipeline_run_id", "goal_type", "query_hash", "source_id",
            name="uq_source_candidates_run_goal_query_source",
        ),
        Index("ix_source_candidates_lookup", "pipeline_run_id", "goal_type", "status"),
    )


class SourceQualityORM(Base):
    """
    Goal-conditioned quality scores for a canonical source.

    Example dimensions:
      - relevance, authority, evidence, depth, accessibility, verifiability, recency

    judge_type examples:
      - heuristic_v1
      - llm
      - hrm
      - sicql
    """
    __tablename__ = "source_quality"

    id = Column(Integer, primary_key=True)

    pipeline_run_id = Column(Integer, nullable=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True)

    goal_type = Column(String(64), nullable=False, index=True)

    dimension = Column(String(64), nullable=False, index=True)
    score = Column(Float, nullable=False)  # 0..1
    weight = Column(Float, nullable=True)  # optional

    judge_type = Column(String(64), nullable=False, server_default="heuristic_v1", index=True)
    judge_version = Column(String(64), nullable=True)

    rationale = Column(Text, nullable=True)
    meta = Column(JSONB, nullable=False, server_default="{}")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "source_id", "goal_type", "dimension", "judge_type",
            name="uq_source_quality_source_goal_dim_judge",
        ),
        Index("ix_source_quality_lookup", "source_id", "goal_type"),
    )
