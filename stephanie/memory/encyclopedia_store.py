from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import func

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.encyclopedia import (ConceptORM, ConceptGemORM,
                                           ConceptQuizORM)

log = logging.getLogger(__name__)


class EncyclopediaStore(BaseSQLAlchemyStore):
    """
    Store for AI Encyclopedia concepts, gems, and quizzes.

    - Follows the same pattern as CaseBookStore:
      each public method wraps a small `op(s)` closure and calls `self._run(op)`.
    - This keeps session lifetime small and avoids long-held locks.
    """

    orm_model = ConceptORM
    default_order_by = ConceptORM.id.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "encyclopedia"

    # ------------------------------------------------------------------
    # Concepts
    # ------------------------------------------------------------------

    def get_concept_by_slug(self, concept_id: str) -> Optional[ConceptORM]:
        def op(s):
            return (
                s.query(ConceptORM)
                .filter(ConceptORM.concept_id == concept_id)
                .one_or_none()
            )

        return self._run(op)

    def get_concept_by_id(self, concept_row_id: int) -> Optional[ConceptORM]:
        def op(s):
            return s.get(ConceptORM, concept_row_id)

        return self._run(op)

    def ensure_concept(
        self,
        concept_id: str,
        name: str,
        summary: str,
        *,
        wiki_url: Optional[str] = None,
        domains: Optional[List[str]] = None,
        sections: Optional[Dict[str, str]] = None,
    ) -> ConceptORM:
        """
        Get or create a concept row.

        - Updates summary/wiki_url/domains/sections if provided and different.
        - Returns the persistent ConceptORM row either way.
        """

        def op(s):
            row = (
                s.query(ConceptORM)
                .filter(ConceptORM.concept_id == concept_id)
                .one_or_none()
            )
            if row:
                updated = False
                if summary and row.summary != summary:
                    row.summary = summary
                    updated = True
                if wiki_url and row.wiki_url != wiki_url:
                    row.wiki_url = wiki_url
                    updated = True
                if domains is not None and row.domains != domains:
                    row.domains = domains
                    updated = True
                if sections is not None:
                    row.sections = sections
                    updated = True
                if updated:
                    s.add(row)
                return row

            row = ConceptORM(
                concept_id=concept_id,
                name=name,
                summary=summary,
                wiki_url=wiki_url,
                domains=domains or [],
                sections=sections,
            )
            s.add(row)
            s.flush()
            return row

        return self._run(op)

    def list_concepts(
        self,
        *,
        domain: Optional[str] = None,
        limit: int = 200,
    ) -> List[ConceptORM]:
        def op(s):
            q = s.query(ConceptORM)
            if domain is not None:
                # domains is JSONB array; use contains([domain]) as with tags
                q = q.filter(ConceptORM.domains.contains([domain]))

            order_col = getattr(ConceptORM, "last_refreshed_at", None)
            if order_col is not None:
                q = q.order_by(order_col.desc())
            else:
                q = q.order_by(ConceptORM.id.desc())
            return q.limit(limit).all()

        return self._run(op)

    def get_frontier_concepts(
        self,
        *,
        min_quiz_total: int = 10,
        min_frontier_count: int = 1,
        min_accuracy: float = 0.3,
        max_accuracy: float = 0.8,
        novelty_max: float = 1.0,
        novelty_min: float = 0.0,
        limit: int = 100,
    ) -> List[ConceptORM]:
        """
        Concepts where:
          - we have enough quiz history
          - accuracy is in the 'interesting' mid-band
          - and there are active frontier quizzes.

        This matches your PretrainZero-style "frontier" band.
        """

        def op(s):
            q = s.query(ConceptORM).filter(
                ConceptORM.quiz_total >= min_quiz_total,
                ConceptORM.frontier_count >= min_frontier_count,
            )
            # guard against NULL accuracy
            q = q.filter(ConceptORM.quiz_accuracy.isnot(None))
            q = q.filter(ConceptORM.quiz_accuracy >= min_accuracy)
            q = q.filter(ConceptORM.quiz_accuracy <= max_accuracy)

            # prioritize those with lots of frontier activity
            q = q.order_by(ConceptORM.frontier_count.desc())
            return q.limit(limit).all()

        return self._run(op)

    # ------------------------------------------------------------------
    # Gems
    # ------------------------------------------------------------------

    def add_gem(
        self,
        concept_id: str,
        text: str,
        *,
        source_type: str,
        source_id: Optional[str] = None,
        source_section: Optional[str] = None,
        source_offset: Optional[int] = None,
        gem_score: Optional[float] = None,
        is_primary: bool = False,
    ) -> ConceptGemORM:
        """
        Attach a gem paragraph to a concept.

        If the concept does not exist, this will fail; call ensure_concept first.
        """

        def op(s):
            concept = (
                s.query(ConceptORM)
                .filter(ConceptORM.concept_id == concept_id)
                .one_or_none()
            )
            if not concept:
                raise ValueError(f"Concept {concept_id!r} does not exist")

            gem = ConceptGemORM(
                concept=concept,
                text=text,
                source_type=source_type,
                source_id=source_id,
                source_section=source_section,
                source_offset=source_offset,
                gem_score=gem_score,
                is_primary=is_primary,
            )
            s.add(gem)
            s.flush()
            return gem

        return self._run(op)

    def get_gems_for_concept(
        self,
        concept_id: str,
        *,
        limit: int = 20,
        primary_first: bool = True,
    ) -> List[ConceptGemORM]:
        def op(s):
            concept = (
                s.query(ConceptORM)
                .filter(ConceptORM.concept_id == concept_id)
                .one_or_none()
            )
            if not concept:
                return []

            q = s.query(ConceptGemORM).filter(
                ConceptGemORM.concept_id_fk == concept.id
            )
            if primary_first:
                q = q.order_by(
                    ConceptGemORM.is_primary.desc(),
                    ConceptGemORM.gem_score.desc().nullslast(),
                    ConceptGemORM.created_at.desc(),
                )
            else:
                q = q.order_by(ConceptGemORM.created_at.desc())
            return q.limit(limit).all()

        return self._run(op)

    # ------------------------------------------------------------------
    # Quizzes
    # ------------------------------------------------------------------

    def record_quiz_result(
        self,
        concept_id: str,
        paragraph_text: str,
        masked_text: str,
        ground_truth_span: str,
        *,
        predicted_span: Optional[str],
        exact_match: bool,
        reward: float,
        is_frontier: bool,
        band_label: Optional[str] = None,
        accuracy_estimate: Optional[float] = None,
    ) -> ConceptQuizORM:
        """
        Insert a ConceptQuizORM row and update aggregate stats on the concept.

        This is the core write-path for PretrainZero-style curriculum stats.
        """

        def op(s):
            concept = (
                s.query(ConceptORM)
                .filter(ConceptORM.concept_id == concept_id)
                .one_or_none()
            )
            if not concept:
                raise ValueError(f"Concept {concept_id!r} does not exist")

            quiz = ConceptQuizORM(
                concept=concept,
                paragraph_text=paragraph_text,
                masked_text=masked_text,
                ground_truth_span=ground_truth_span,
                predicted_span=predicted_span,
                exact_match=exact_match,
                reward=reward,
                is_frontier=is_frontier,
                accuracy_estimate=accuracy_estimate,
            )
            s.add(quiz)

            # update aggregate stats on the concept row
            concept.update_quiz_stats(
                total_delta=1,
                correct_delta=1 if exact_match else 0,
                frontier_delta=1 if is_frontier else 0,
                band_label=band_label,
            )
            s.add(concept)

            s.flush()
            return quiz

        return self._run(op)

    def recompute_quiz_stats(self, concept_id: str) -> Optional[ConceptORM]:
        """
        Slow but precise: recompute quiz stats from ai_concept_quizzes
        for one concept. Useful if you ever need to repair stats.
        """

        def op(s):
            concept = (
                s.query(ConceptORM)
                .filter(ConceptORM.concept_id == concept_id)
                .one_or_none()
            )
            if not concept:
                return None

            q = s.query(
                func.count(ConceptQuizORM.id),
                func.sum(func.coalesce(func.cast(ConceptQuizORM.exact_match, int), 0)),
                func.sum(
                    func.case(
                        (ConceptQuizORM.is_frontier.is_(True), 1), else_=0
                    )
                ),
            ).filter(ConceptQuizORM.concept_id_fk == concept.id)

            total, correct, frontier = q.one()
            total = int(total or 0)
            correct = int(correct or 0)
            frontier = int(frontier or 0)

            concept.quiz_total = total
            concept.quiz_correct = correct
            concept.frontier_count = frontier
            if total > 0:
                concept.quiz_accuracy = float(correct) / float(total)
            else:
                concept.quiz_accuracy = None

            s.add(concept)
            s.flush()
            return concept

        return self._run(op)
