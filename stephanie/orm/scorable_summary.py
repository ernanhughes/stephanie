# stephanie/orm/scorable_summary.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (JSON, Column, DateTime, Float, Index, Integer, String,
                        Text)

from stephanie.orm.base import Base


class ScorableSummaryORM(Base):
    """
    Generic summary record for any Scorable-like thing.

    We do NOT FK to a specific table; instead we store:
      - scorable_type: e.g. "document", "section", "chat_turn", ...
      - scorable_id:   integer or string id of that scorable

    This lets you attach multiple summaries (different tools/models/versions)
    to any scorable object without schema churn.
    """

    __tablename__ = "scorable_summaries"

    id = Column(Integer, primary_key=True)

    # Generic link to "whatever this summary is for"
    scorable_type = Column(String(64), nullable=False, index=True)
    scorable_id = Column(String(64), nullable=False, index=True)

    # Optional: run / pipeline info
    run_id = Column(String(64), nullable=True, index=True)

    # Provenance
    tool_name = Column(String(128), nullable=False)   # e.g. "section_summarizer"
    summary_kind = Column(String(32), nullable=True)  # "section", "doc", "title", ...
    model_name = Column(String(256), nullable=False)  # HF model id
    model_version = Column(String(64), nullable=True) # your own versioning, if any
    prompt_hash = Column(String(64), nullable=True)   # if you hash the prompt template

    # Content
    title = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)

    # Basic metrics
    source_char_len = Column(Integer, nullable=True)
    summary_char_len = Column(Integer, nullable=True)
    compression_ratio = Column(Float, nullable=True)  # summary_len / source_len

    # Human + auto quality
    quality_label = Column(Integer, nullable=True)    # e.g. 1–5, 0–100, etc.
    auto_quality_score = Column(Float, nullable=True) # from TinyCritic/SICQL/etc

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Free-form extras (e.g. entropy stats, Vibe features, etc.)
    meta = Column(JSON, nullable=True)

    __table_args__ = (
        Index(
            "ix_scorable_summaries_scorable",
            "scorable_type",
            "scorable_id",
        ),
    )

    # Convenience constructor
    @classmethod
    def from_scorable(
        cls,
        scorable_type: str,
        scorable_id: str,
        *,
        tool_name: str,
        model_name: str,
        title: str,
        summary: str,
        run_id: Optional[str] = None,
        summary_kind: Optional[str] = None,
        model_version: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        source_text: Optional[str] = None,
        quality_label: Optional[int] = None,
        auto_quality_score: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ScorableSummaryORM":
        source_len = len(source_text) if source_text is not None else None
        summary_len = len(summary) if summary is not None else None
        compression = (
            float(summary_len) / float(source_len)
            if (source_len and summary_len and source_len > 0)
            else None
        )

        return cls(
            scorable_type=scorable_type,
            scorable_id=str(scorable_id),
            run_id=run_id,
            tool_name=tool_name,
            summary_kind=summary_kind,
            model_name=model_name,
            model_version=model_version,
            prompt_hash=prompt_hash,
            title=title,
            summary=summary,
            source_char_len=source_len,
            summary_char_len=summary_len,
            compression_ratio=compression,
            quality_label=quality_label,
            auto_quality_score=auto_quality_score,
            meta=meta or {},
        )
