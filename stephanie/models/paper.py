# stephanie/models/paper.py
from __future__ import annotations

from sqlalchemy import (BigInteger, Column, DateTime, Float, ForeignKey, Index,
                        Integer, String, Text, UniqueConstraint)
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

    sections = relationship(
        "PaperSectionORM",
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

class PaperSectionORM(Base):
    __tablename__ = "paper_sections"
    
    id = Column(String, primary_key=True)  # e.g. "2505.08827::parse-v3::1.2.4"
    paper_id = Column(String, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(String, ForeignKey("paper_runs.id", ondelete="CASCADE"), nullable=True, index=True)  # ✅ ADDED

    # Hierarchy
    parent_id = Column(String, ForeignKey("paper_sections.id", ondelete="CASCADE"), nullable=True, index=True)
    level = Column(Integer, nullable=False, default=0)     # 0 = top, 1 = subsection, ...
    path = Column(String, nullable=False)                   # e.g. "1", "2.3", "4.1.5" — ✅ stable anchor

    # Position in original doc
    section_index = Column(Integer, nullable=False)        # flat index within this run (for fallback)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    start_page = Column(Integer, nullable=True)
    end_page = Column(Integer, nullable=True)

    # Content
    text = Column(Text, nullable=True)
    title = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    content_hash = Column(String, nullable=True, index=True)  # SHA256 of normalized(text)
    token_count = Column(Integer, nullable=True)              # helps budgeting

    # Metadata
    meta = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    paper = relationship("PaperORM", back_populates="sections")
    parent = relationship("PaperSectionORM", remote_side=[id], backref="children")

    __table_args__ = (
        UniqueConstraint("paper_id", "run_id", "path", name="uq_paper_run_path"),
        Index("ix_paper_sections_paper_run", "paper_id", "run_id"),
        Index("ix_paper_sections_parent", "parent_id"),
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

    def to_dict(self) -> dict:
        return {
            "ref_arxiv_id": self.ref_arxiv_id,
            "doi": self.doi,
            "title": self.title,
            "year": self.year,
            "url": self.url,
            "raw_citation": self.raw_citation,
            "source": self.source,
            "raw": self.raw,
        }   


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

    id = Column(String, primary_key=True)  # UUID hex
    paper_id = Column(
        String,
        ForeignKey("papers.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    run_type = Column(String, nullable=False)     # "blog", "parse", "score", "summary"
    variant = Column(String, nullable=True)       # "v1", "with_diagrams", etc.

    # Core config + state
    config = Column(JSONB, nullable=False)      # immutable input config (serialized)
    meta = Column(JSONB, nullable=True)         # mutable runtime metadata (e.g., hashes, version, env)
    stats = Column(JSONB, nullable=True)        # counters, graph nodes, etc.

    artifact_path = Column(String, nullable=True)

    # Lifecycle
    status = Column(String, nullable=False, default="started")  # "started", "ok", "error", "canceled"
    error = Column(Text, nullable=True)

    # Evaluation / judging
    ai_score = Column(Float, nullable=True)           # scalar fallback (e.g., for ranking)
    ai_scores = Column(JSONB, nullable=True)          # {"clarity": 0.92, "novelty": 0.78}
    ai_rationale = Column(Text, nullable=True)        # multi-line rationale
    judge_meta = Column(JSONB, nullable=True)         # {"model": "gpt-4o", "prompt_hash": "...", "calibration": "v2"}

    # Provenance & debugging
    duration_ms = Column(Integer, nullable=True)      # optional but very useful
    prompt_hash = Column(String, nullable=True)       # hash of full prompt
    code_version = Column(String, nullable=True)      # e.g., git commit or semver

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class PaperRunFeatureORM(Base):
    __tablename__ = "paper_run_features"

    id = Column(String, primary_key=True)  # UUID or deterministic hash

    run_id = Column(
        String,
        ForeignKey("paper_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    paper_id = Column(
        String,
        ForeignKey("papers.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )  

    extractor = Column(String, nullable=False)   # "v1_basic", "v2_semantic_chunks", "code_extractor"
    features = Column(JSONB, nullable=False)     # full dict — e.g., {"section_count": 8, "avg_token_len": 142}

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("run_id", "extractor", name="uq_run_extractor"),
        Index("ix_paper_run_features_paper", "paper_id"),
    )


class PaperRunComparisonORM(Base):
    __tablename__ = "paper_run_comparisons"

    id = Column(String, primary_key=True)  # UUID hex

    paper_id = Column(
        String,
        ForeignKey("papers.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    left_run_id = Column(
        String,
        ForeignKey("paper_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    right_run_id = Column(
        String,
        ForeignKey("paper_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    winner_run_id = Column(
        String,
        ForeignKey("paper_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    judge_source = Column(String, nullable=False, default="llm")  # "human", "llm", "ebt", "mrq", "ensemble"
    preference = Column(Float, nullable=False)  # +1 = left preferred, -1 = right preferred, 0 = tie

    scores = Column(JSONB, nullable=True)     # {"margin": 0.32, "clarity_diff": 0.1, ...}
    rationale = Column(Text, nullable=True)
    meta = Column(JSONB, nullable=True)       # calibration, model version, prompt hash, etc.

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_paper_run_comparisons_paper", "paper_id"),
        Index("ix_paper_run_comparisons_runs", "left_run_id", "right_run_id"),
        Index("ix_paper_run_comparisons_winner", "winner_run_id"),
    )

class PaperRunEventORM(Base):
    __tablename__ = "paper_run_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        String,
        ForeignKey("paper_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # Context
    stage = Column(String, nullable=True)        # "paper_import", "blog_gen", "scoring"
    
    # Event core
    event_type = Column(String, nullable=False)  # "info"|"metric"|"warn"|"error"|"trace"
    message = Column(Text, nullable=True)
    
    # Structured payload
    data = Column(JSONB, nullable=True)          # arbitrary — e.g., tokens used, latency, model ID
    payload = Column(JSONB, nullable=True)       # legacy alias (keep for compat; prefer `data`)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_paper_run_events_run_stage", "run_id", "stage"),
        Index("ix_paper_run_events_type", "event_type"),
    )

class PaperArtifactORM(Base):
    __tablename__ = "paper_artifacts"

    id = Column(String, primary_key=True)  # UUID or `"paper::run::type::idx"`
    paper_id = Column(String, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(String, ForeignKey("paper_runs.id", ondelete="CASCADE"), nullable=True, index=True)
    section_id = Column(String, ForeignKey("paper_sections.id", ondelete="SET NULL"), nullable=True, index=True)

    artifact_type = Column(String, nullable=False)  # "figure", "table", "equation", "code", "image", "diagram"
    page_number = Column(Integer, nullable=True)
    bbox = Column(JSONB, nullable=True)  # {"x0": 100, "y0": 200, "x1": 300, "y1": 400}

    caption = Column(Text, nullable=True)
    alt_text = Column(Text, nullable=True)

    storage_path = Column(String, nullable=True)   # local/remote path
    blob_ref = Column(String, nullable=True)       # e.g., S3 key, DB blob ID
    content_hash = Column(String, nullable=True, index=True)

    meta = Column(JSONB, nullable=True)  # OCR result, LaTeX source, model used, etc.

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_paper_artifacts_paper_type", "paper_id", "artifact_type"),
        Index("ix_paper_artifacts_section", "section_id"),
    ) 

class PaperSectionScoreORM(Base):
    __tablename__ = "paper_section_scores"

    id = Column(String, primary_key=True)  # UUID
    section_id = Column(String, ForeignKey("paper_sections.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(String, ForeignKey("paper_runs.id", ondelete="SET NULL"), nullable=True, index=True)

    dimension = Column(String, nullable=False)  # "clarity", "novelty", "risk", "relevance"
    source = Column(String, nullable=False)     # "llm", "ebt", "human", "svm"
    score = Column(Float, nullable=False)       # 0.0–1.0 or -1.0–1.0
    rationale = Column(Text, nullable=True)

    meta = Column(JSONB, nullable=True)  # prompt, model, calibration, etc.

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("section_id", "dimension", "source", name="uq_section_dim_source"),
        Index("ix_paper_section_scores_section", "section_id"),
        Index("ix_paper_section_scores_run", "run_id"),
    )

class TrainingSourceLinkORM(Base):
    __tablename__ = "training_source_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_event_id = Column(String, nullable=False, index=True)  # your event ID (e.g., from RLHF pipeline)

    source_type = Column(String, nullable=False)  # "paper", "paper_section", "paper_run", "blog_draft"
    source_id = Column(String, nullable=False, index=True)         # e.g., section.id

    meta = Column(JSONB, nullable=True)  # weight, extraction_version, filter_reason, etc.

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_training_source_links_event", "training_event_id"),
        Index("ix_training_source_links_source", "source_type", "source_id"),
    )
    


class PaperDoclingPageORM(Base):
    """
    Per-page DocTags cache from SmolDocling (or other doc-to-tags backends).
    This is the canonical conversion artifact that downstream tools can re-parse.
    """
    __tablename__ = "paper_docling_pages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(String, ForeignKey("paper_runs.id", ondelete="CASCADE"), nullable=True, index=True)

    page_num = Column(Integer, nullable=False)      # 1-based
    model_name = Column(String, nullable=True)
    dpi = Column(Integer, nullable=True)

    doctags = Column(Text, nullable=False)          # raw doctags string
    meta = Column(JSONB, nullable=True)             # warnings, bbox flags, parse stats, etc.

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("paper_id", "run_id", "page_num", name="uq_docling_paper_run_page"),
        Index("ix_docling_pages_paper_run", "paper_id", "run_id"),
    )
