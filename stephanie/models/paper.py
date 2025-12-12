# stephanie/models/paper.py
from __future__ import annotations

from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, Index, Float
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import LargeBinary

from stephanie.models.base import Base


class PaperORM(Base):
    """
    Canonical cached representation of a paper (usually arXiv).
    Primary key is the stable key you use in the graph: arxiv_id or local stem.
    """
    __tablename__ = "papers"

    id = Column(String, primary_key=True)                 # e.g. "2505.08827" or local key
    source = Column(String, nullable=False, default="arxiv")  # "arxiv" | "local" | "hf" | ...

    url = Column(String, nullable=True, index=True)
    title = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    authors = Column(ARRAY(String), nullable=True)
    published = Column(String, nullable=True)             # ISO string (matches your fetch_arxiv_metadata)

    text = Column(Text, nullable=True)
    text_hash = Column(String, nullable=True, index=True)

    pdf_path = Column(String, nullable=True)
    pdf_bytes = Column(LargeBinary, nullable=True)        # optional: store the binary
    pdf_sha256 = Column(String, nullable=True, index=True)

    meta = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    references = relationship(
        "PaperReferenceORM",
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    ) 

    similars = relationship(
        "PaperSimilarORM",
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    blog_runs = relationship(
        "PaperBlogRunORM",
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )



class PaperReferenceORM(Base):
    __tablename__ = "paper_references"

    id = Column(Integer, primary_key=True)
    paper_id = Column(String, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)

    order_idx = Column(Integer, nullable=True)

    ref_arxiv_id = Column(String, nullable=True, index=True)
    doi = Column(String, nullable=True, index=True)
    title = Column(Text, nullable=True)
    year = Column(Integer, nullable=True, index=True)
    url = Column(String, nullable=True)
    raw_citation = Column(Text, nullable=True)

    source = Column(String, nullable=False, default="parsed_pdf")  # "parsed_pdf" | "openalex" | ...
    raw = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    paper = relationship("PaperORM", back_populates="references")

    __table_args__ = (
        # prevent “same ref inserted 100x”
        UniqueConstraint("paper_id", "order_idx", name="uq_paper_ref_order"),
        Index("ix_paper_references_paper_ref", "paper_id", "ref_arxiv_id"),
    )


class PaperSimilarORM(Base):
    __tablename__ = "paper_similar"

    id = Column(Integer, primary_key=True)
    paper_id = Column(String, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)

    provider = Column(String, nullable=False, default="hf_similar")
    rank = Column(Integer, nullable=True)
    score = Column(Float, nullable=True)

    similar_arxiv_id = Column(String, nullable=False, index=True)
    url = Column(String, nullable=True)
    title = Column(Text, nullable=True)

    raw = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    paper = relationship("PaperORM", back_populates="similars")

    __table_args__ = (
        UniqueConstraint("paper_id", "provider", "similar_arxiv_id", name="uq_paper_similar_unique"),
        Index("ix_paper_similar_paper_rank", "paper_id", "rank"),
    )

class PaperRunORM(Base):
    __tablename__ = "paper_runs"

    id = Column(String, primary_key=True)  # your universal ID
    paper_id = Column(
        String,
        ForeignKey("papers.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    run_type = Column(String, nullable=False)  
    # e.g. "blog", "analysis", "summary"

    config = Column(JSONB, nullable=False)      # BlogConfig dump
    stats = Column(JSONB, nullable=True)        # graph stats, section counts, etc.

    artifact_path = Column(String, nullable=True)  # markdown / output path

    ai_score = Column(Float, nullable=True)
    ai_rationale = Column(String, nullable=True)
    ai_judge = Column(String, nullable=True)
    ai_prompt_hash = Column(String, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class PaperRunFeatureORM(Base):
    """
    Stores extracted features for a paper run.
    Features are stored as JSONB to allow flexible evolution.
    """
    __tablename__ = "paper_run_features"

    id = Column(String, primary_key=True)
    run_id = Column(
        String,
        ForeignKey("paper_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    features = Column(JSONB, nullable=False)
    extractor = Column(String, nullable=True)      # e.g. "v1_basic"
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class PaperRunComparisonORM(Base):
    """
    Pairwise comparison between two runs of the same paper.
    Used for ranking / preference learning.
    """
    __tablename__ = "paper_run_comparisons"

    id = Column(String, primary_key=True)

    paper_id = Column(String, index=True, nullable=False)

    run_a_id = Column(
        String,
        ForeignKey("paper_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_b_id = Column(
        String,
        ForeignKey("paper_runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    # preference: +1 means A preferred, -1 means B preferred
    preference = Column(Float, nullable=False)

    judge = Column(String, nullable=True)
    rationale = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PaperRunEventORM(Base):
    """
    Step-level trace events for a paper run.
    Used for debugging, analysis, and future HRM / RL training.
    """
    __tablename__ = "paper_run_events"

    id = Column(String, primary_key=True)
    run_id = Column(
        String,
        ForeignKey("paper_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    event_type = Column(String, nullable=False)   # e.g. "retrieve", "score", "generate"
    payload = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
