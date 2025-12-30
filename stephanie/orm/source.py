from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, Text, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from stephanie.orm.base import Base


class SourceORM(Base):
    """
    Canonical inbound provenance object.
    Examples:
      - url: https://arxiv.org/pdf/2506.21734.pdf
      - file: C:\\AI\\papers\\2506.21734.pdf
      - db:   postgres://.../table/key
      - ai:   llm://model/prompt_hash (generated)
    """
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True)

    # url|file|db|ai|generated|unknown
    source_type = Column(Text, nullable=False, index=True)

    # how to retrieve it
    source_uri = Column(Text, nullable=False, index=True)
    canonical_uri = Column(Text, nullable=True, index=True)

    # optional display fields
    title = Column(Text, nullable=True)
    snippet = Column(Text, nullable=True)

    # content identity (if fetched)
    content_hash = Column(Text, nullable=True, index=True)
    mime_type = Column(Text, nullable=True)

    # coarse trust/quality priors (can be overridden per-goal in SourceQuality)
    trust_score = Column(Float, nullable=True)     # 0..1
    quality_score = Column(Float, nullable=True)   # 0..1
    verification = Column(Text, nullable=True, index=True)  # verified|unverified|generated|unknown

    meta = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("source_type", "source_uri", name="uq_sources_type_uri"),
        Index("ix_sources_verification", "verification"),
    )


class SourceCandidateORM(Base):
    """
    A source as it appeared in a particular run/query result set (search hit).
    This is what lets you reproduce “why was this source considered?”
    """
    __tablename__ = "source_candidates"

    id = Column(Integer, primary_key=True)

    pipeline_run_id = Column(Integer, nullable=False, index=True)

    # query that produced this hit (store raw + hash)
    query_text = Column(Text, nullable=False)
    query_hash = Column(Text, nullable=False, index=True)

    source_id = Column(Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True)

    # where it came from: "arxiv", "web", "wikipedia", "hf", etc.
    provider = Column(Text, nullable=True, index=True)

    # "paper" | "blog" | "repo" | "wiki" | "pdf" | ...
    result_type = Column(Text, nullable=True, index=True)

    rank = Column(Integer, nullable=True)
    score = Column(Float, nullable=True)  # provider score if available

    meta = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("pipeline_run_id", "query_hash", "source_id", name="uq_source_candidates_run_query_source"),
        Index("ix_source_candidates_run_provider", "pipeline_run_id", "provider"),
    )


class SourceQualityORM(Base):
    """
    Goal-conditioned quality assessment for a source.
    Example:
      - goal_type="research" => arXiv gets high trust, random blog gets lower
      - goal_type="marketing" might weight differently
    """
    __tablename__ = "source_quality"

    id = Column(Integer, primary_key=True)

    pipeline_run_id = Column(Integer, nullable=True, index=True)  # optional: per-run
    goal_type = Column(Text, nullable=False, index=True)

    source_id = Column(Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True)

    trust_score = Column(Float, nullable=True)     # 0..1
    quality_score = Column(Float, nullable=True)   # 0..1
    verification = Column(Text, nullable=True)     # verified|unverified|generated|unknown

    method = Column(Text, nullable=False, server_default="heuristic", index=True)
    rationale = Column(Text, nullable=True)

    meta = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("goal_type", "source_id", "method", "pipeline_run_id", name="uq_source_quality_goal_source_method_run"),
        Index("ix_source_quality_goal", "goal_type"),
    )
