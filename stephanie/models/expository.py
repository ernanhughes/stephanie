# stephanie/models/expository.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Index, Integer,
                        String, Text)

from stephanie.db.base import Base


class ExpositorySnippet(Base):
    __tablename__ = "expository_snippets"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, index=True)             # your Document table id
    section = Column(String(256), index=True)
    order_idx = Column(Integer, default=0)
    text = Column(Text, nullable=False)
    features = Column(JSON, default={})              # section_cue, rhet_cues, cite_density, novelty_lex, xref_density, symbol_density, len_tokens, readability
    expository_score = Column(Float, index=True)
    bloggability_score = Column(Float, index=True)
    picked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)

Index("ix_expo_doc_section", ExpositorySnippet.doc_id, ExpositorySnippet.section)

class ExpositoryBuffer(Base):
    __tablename__ = "expository_buffers"
    id = Column(Integer, primary_key=True)
    topic = Column(String(512), index=True)
    snippet_ids = Column(JSON, default=list)         # [int, ...]
    meta = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

class BlogDraft(Base):
    __tablename__ = "blog_drafts"
    id = Column(Integer, primary_key=True)
    topic = Column(String(512), index=True)
    source_snippet_ids = Column(JSON, default=list)
    draft_md = Column(Text)
    arena_passes = Column(Integer, default=0)
    readability = Column(Float, index=True)
    local_coherence = Column(Float, index=True)
    repetition_penalty = Column(Float, default=0.0)
    kept = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class PaperSourceQueueORM(Base):
    __tablename__ = "paper_source_queue"
    id = Column(Integer, primary_key=True)
    topic = Column(String(256), index=True)
    url = Column(Text, nullable=False, unique=True)
    source = Column(String(128), default="manual")    # e.g., recommend_similar_papers
    status = Column(String(32), default="pending")    # pending|fetched|parsed|failed|skipped
    meta = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

Index("ix_psq_topic_status", PaperSourceQueueORM.topic, PaperSourceQueueORM.status)
