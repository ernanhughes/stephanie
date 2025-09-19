from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Optional

import numpy as np

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.goal import GoalORM
from stephanie.models.hypothesis import HypothesisORM


class HypothesisStore(BaseSQLAlchemyStore):
    orm_model = HypothesisORM
    default_order_by = HypothesisORM.created_at.desc()
    
    def __init__(self, session_maker, logger=None, embedding_store=None):
        super().__init__(session_maker, logger)
        self.embedding_store = embedding_store  # Optional embedding model
        self.name = "hypotheses"

    def name(self) -> str:
        return self.name

    # -------------------
    # Insert / Update
    # -------------------
    def insert(self, hypothesis: HypothesisORM) -> int:
        """Insert a new hypothesis into the database."""
        def op(s):
            s.add(hypothesis)
            s.flush()
            if self.logger:
                self.logger.log(
                    "HypothesisInserted",
                    {
                        "hypothesis_id": hypothesis.id,
                        "goal_id": hypothesis.goal_id,
                        "strategy": hypothesis.strategy,
                        "timestamp": hypothesis.created_at.isoformat(),
                    },
                )
            return hypothesis.id
        return self._run(op)

    def update_review(self, hyp_id: int, review: str) -> None:
        def op(s):
            hyp = s.get(HypothesisORM, hyp_id)
            if not hyp:
                raise ValueError(f"No hypothesis found with ID {hyp_id}")
            hyp.review = review
            if self.logger:
                self.logger.log(
                    "ReviewStored",
                    {"hypothesis_id": hyp_id, "review_snippet": (review or '')[:100]},
                )
        self._run(op)

    def update_reflection(self, hyp_id: int, reflection: str) -> None:
        def op(s):
            hyp = s.get(HypothesisORM, hyp_id)
            if not hyp:
                raise ValueError(f"No hypothesis found with ID {hyp_id}")
            hyp.reflection = reflection
            if self.logger:
                self.logger.log(
                    "ReflectionStored",
                    {"hypothesis_id": hyp_id, "reflection_snippet": (reflection or '')[:100]},
                )
        self._run(op)

    def update_elo_rating(self, hyp_id: int, new_rating: float) -> None:
        def op(s):
            hyp = s.get(HypothesisORM, hyp_id)
            if not hyp:
                raise ValueError(f"No hypothesis found with ID {hyp_id}")
            hyp.elo_rating = new_rating
            if self.logger:
                self.logger.log(
                    "HypothesisEloUpdated",
                    {"hypothesis_id": hyp_id, "elo_rating": new_rating},
                )
        self._run(op)

    def soft_delete(self, hyp_id: int) -> None:
        """Soft-delete a hypothesis (set enabled = False)."""
        def op(s):
            hyp = s.get(HypothesisORM, hyp_id)
            if not hyp:
                raise ValueError(f"No hypothesis found with ID {hyp_id}")
            hyp.enabled = False
            if self.logger:
                self.logger.log("HypothesisSoftDeleted", {"hypothesis_id": hyp_id})
        self._run(op)

    # -------------------
    # Retrieval
    # -------------------
    def get_by_goal(self, goal_text: str, limit: int = 10, source=None) -> List[HypothesisORM]:
        def op(s):
            q = s.query(HypothesisORM).join(GoalORM).filter(GoalORM.goal_text == goal_text)
            if source:
                from stephanie.models import EvaluationORM
                q = q.join(EvaluationORM).filter(EvaluationORM.source == source)
            return q.limit(limit).all()
        return self._run(op)

    def get_latest(self, goal_text: str, limit: int = 10) -> List[HypothesisORM]:
        def op(s):
            return (
                s.query(HypothesisORM)
                .join(GoalORM)
                .filter(GoalORM.goal_text == goal_text)
                .order_by(HypothesisORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def get_unreflected(self, goal_text: str, limit: int = 10) -> List[HypothesisORM]:
        def op(s):
            return (
                s.query(HypothesisORM)
                .join(GoalORM)
                .filter(GoalORM.goal_text == goal_text, HypothesisORM.reflection.is_(None))
                .limit(limit)
                .all()
            )
        return self._run(op)

    def get_unreviewed(self, goal_text: str, limit: int = 10) -> List[HypothesisORM]:
        def op(s):
            return (
                s.query(HypothesisORM)
                .join(GoalORM)
                .filter(GoalORM.goal_text == goal_text, HypothesisORM.review.is_(None))
                .limit(limit)
                .all()
            )
        return self._run(op)

    def get_from_text(self, query: str, threshold: float = 0.95) -> Optional[HypothesisORM]:
        """Exact or fuzzy match for hypothesis text."""
        def op(s):
            result = s.query(HypothesisORM).filter(HypothesisORM.text == query).first()
            if result:
                return result
            result = s.query(HypothesisORM).filter(HypothesisORM.text.ilike(f"%{query}%")).first()
            if result and result.text:
                sim = SequenceMatcher(None, result.text, query).ratio()
                if sim >= threshold:
                    return result
            return None
        return self._run(op)

    def get_by_id(self, hyp_id: int) -> Optional[HypothesisORM]:
        return self._run(lambda s: s.get(HypothesisORM, hyp_id))

    def get_all(self, limit: int = 100) -> List[HypothesisORM]:
        def op(s):
            return (
                s.query(HypothesisORM)
                .order_by(HypothesisORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def get_similar(self, query: str, limit: int = 3) -> List[str]:
        """
        Semantic similarity search using embeddings (if available).
        Requires pgvector or similar backend in embedding_store.
        """
        if not self.embedding_store:
            return []

        try:
            query_embedding = self.embedding_store.get_or_create(query)
            with self.embedding_store.conn.cursor() as cur:
                cur.execute(
                    "SELECT text FROM hypotheses ORDER BY embedding <-> %s LIMIT %s",
                    (np.array(query_embedding), limit),
                )
                results = [row[0] for row in cur.fetchall()]
            if self.logger:
                self.logger.log(
                    "SimilarHypothesesFound",
                    {"query": query[:100], "matches": [r[:100] for r in results]},
                )
            return results
        except Exception as e:
            if self.logger:
                self.logger.log("SimilarHypothesesSearchFailed", {"error": str(e)})
            return []
