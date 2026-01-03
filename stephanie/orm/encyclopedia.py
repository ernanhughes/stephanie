# stephanie/orm/encyclopedia.py

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Index, Integer, String, Text, UniqueConstraint)
# If you already have a central Base, import that instead:
# from stephanie.db.base import Base
from sqlalchemy.orm import (Mapped, declarative_base, mapped_column,
                            relationship)

Base = declarative_base()


class ConceptORM(Base):
    """
    Canonical AI Encyclopedia concept.

    Dual role:
      - Human-facing knowledge: name, summary, wiki info, linked gems.
      - Curriculum-facing stats: quiz counts, accuracy, frontier stats.

    concept_id should match the slug you use in the KG (e.g., 'cross_entropy').
    """
    __tablename__ = "ai_concepts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Stable external identifier (e.g. Wikipedia slug or your own)
    concept_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    # Human-facing
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    wiki_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Optional: full structured sections (e.g., {"Definition": "...", "History": "..."})
    sections: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON, nullable=True)

    # Tags/domains e.g. ["ML", "RL", "Optimization"]
    domains: Mapped[List[str]] = mapped_column(JSON, default=list)

    # ----- Curriculum / PretrainZero-style stats -----

    # Total number of quizzes run against any span tied to this concept
    quiz_total: Mapped[int] = mapped_column(Integer, default=0)

    # Total number of exact-match successes (can compute accuracy from this)
    quiz_correct: Mapped[int] = mapped_column(Integer, default=0)

    # Cached accuracy (quiz_correct / quiz_total) for fast querying
    quiz_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Number of quizzes currently considered in the "frontier band"
    # (e.g., 30–80% accuracy at quiz level)
    frontier_count: Mapped[int] = mapped_column(Integer, default=0)

    # Optional richer histogram, e.g. {"0-0.3": n, "0.3-0.8": m, "0.8-1.0": k}
    quiz_histogram: Mapped[Optional[Dict[str, int]]] = mapped_column(
        JSON, nullable=True
    )

    last_quiz_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_refreshed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    # Relationships
    gems: Mapped[List["ConceptGemORM"]] = relationship(
        "ConceptGemORM",
        back_populates="concept",
        cascade="all, delete-orphan",
    )
    quizzes: Mapped[List["ConceptQuizORM"]] = relationship(
        "ConceptQuizORM",
        back_populates="concept",
        cascade="all, delete-orphan",
    )

    def update_quiz_stats(
        self,
        total_delta: int,
        correct_delta: int,
        frontier_delta: int,
        band_label: Optional[str] = None,
    ) -> None:
        """Convenience method to keep stats consistent."""
        self.quiz_total += int(total_delta)
        self.quiz_correct += int(correct_delta)
        self.frontier_count += int(frontier_delta)

        if self.quiz_total > 0:
            self.quiz_accuracy = float(self.quiz_correct) / float(self.quiz_total)
        else:
            self.quiz_accuracy = None

        if band_label:
            hist = self.quiz_histogram or {}
            hist[band_label] = int(hist.get(band_label, 0)) + 1
            self.quiz_histogram = hist

        self.last_quiz_at = datetime.utcnow()


class ConceptGemORM(Base):
    """
    A 'gem' paragraph: high-quality explanation or insight from a paper, blog, etc.
    Linked to a concept and back to the original source.
    """
    __tablename__ = "ai_concept_gems"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id_fk: Mapped[int] = mapped_column(
        Integer, ForeignKey("ai_concepts.id", ondelete="CASCADE"), index=True
    )

    # Basic text and metadata
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # e.g. "paper", "blog", "code", "wiki"
    source_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # e.g. paper arxiv ID, URL hash
    source_section: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_offset: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Optional quality / selection scores
    gem_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    concept: Mapped[ConceptORM] = relationship("ConceptORM", back_populates="gems")

    __table_args__ = (
        # Avoid accidental duplicates from repeated ingestion
        UniqueConstraint(
            "concept_id_fk", "source_type", "source_id", "source_offset",
            name="uq_concept_gem_source",
        ),
    )


class ConceptQuizORM(Base):
    """
    A single quiz instance over a span associated with a concept.

    You can store granular info here and aggregate onto ConceptORM.quiz_* columns.
    """
    __tablename__ = "ai_concept_quizzes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id_fk: Mapped[int] = mapped_column(
        Integer, ForeignKey("ai_concepts.id", ondelete="CASCADE"), index=True
    )

    # The original paragraph and the masked version
    paragraph_text: Mapped[str] = mapped_column(Text, nullable=False)
    masked_text: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth_span: Mapped[str] = mapped_column(Text, nullable=False)

    # Result of one roll-out
    predicted_span: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    exact_match: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Is this quiz currently considered in the frontier band for this concept?
    is_frontier: Mapped[bool] = mapped_column(Boolean, default=False)

    # Optional: store accuracy at the time it was evaluated
    accuracy_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    concept: Mapped[ConceptORM] = relationship("ConceptORM", back_populates="quizzes")

    __table_args__ = (
        Index("ix_concept_quiz_frontier", "concept_id_fk", "is_frontier"),
    )
