# stores/hypothesis_store.py
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from co_ai.models.goal import GoalORM
from co_ai.models.hypothesis import HypothesisORM


class HypothesisStore:
    def __init__(self, session: Session, logger=None, embedding_store=None):
        self.session = session
        self.logger = logger
        self.embedding_store = embedding_store  # Optional embedding model
        self.name = "hypotheses"
    
    def name(self) -> str:
        return "hypotheses"

    def insert(self, hypothesis: HypothesisORM) -> int:
        """
        Inserts a new hypothesis into the database.
        Assumes goal and prompt are already resolved to IDs.
        """
        try:
            self.session.add(hypothesis)
            self.session.flush()  # To get ID before commit

            if self.logger:
                self.logger.log("HypothesisInserted", {
                    "hypothesis_id": hypothesis.id,
                    "goal_id": hypothesis.goal_id,
                    "strategy": hypothesis.strategy,
                    "length": len(hypothesis.text),
                    "timestamp": hypothesis.created_at.isoformat()
                })

            return hypothesis.id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("HypothesisInsertFailed", {"error": str(e)})
            raise

    def update_review(self, hyp_id: int, review: str):
        """
        Updates the review field for a hypothesis.
        """
        hyp = self.session.query(HypothesisORM).get(hyp_id)
        if not hyp:
            raise ValueError(f"No hypothesis found with ID {hyp_id}")

        hyp.review = review
        self.session.commit()

        if self.logger:
            self.logger.log("ReviewStored", {
                "hypothesis_id": hyp_id,
                "review_snippet": (review or "")[:100]
            })

    def update_reflection(self, hyp_id: int, reflection: str):
        """
        Updates the reflection field for a hypothesis.
        """
        hyp = self.session.query(HypothesisORM).get(hyp_id)
        if not hyp:
            raise ValueError(f"No hypothesis found with ID {hyp_id}")

        hyp.reflection = reflection
        self.session.commit()

        if self.logger:
            self.logger.log("ReflectionStored", {
                "hypothesis_id": hyp_id,
                "reflection_snippet": (reflection or "")[:100]
            })

    def update_elo_rating(self, hyp_id: int, new_rating: float):
        """
        Updates the ELO rating of a hypothesis after pairwise comparison.
        """
        hyp = self.session.query(HypothesisORM).get(hyp_id)
        if not hyp:
            raise ValueError(f"No hypothesis found with ID {hyp_id}")

        hyp.elo_rating = new_rating
        self.session.commit()

        if self.logger:
            self.logger.log("HypothesisEloUpdated", {
                "hypothesis_id": hyp_id,
                "elo_rating": new_rating
            })

    def soft_delete(self, hyp_id: int):
        """
        Soft-deletes a hypothesis by setting enabled = False
        """
        hyp = self.session.query(HypothesisORM).get(hyp_id)
        if not hyp:
            raise ValueError(f"No hypothesis found with ID {hyp_id}")

        hyp.enabled = False
        self.session.commit()

        if self.logger:
            self.logger.log("HypothesisSoftDeleted", {"hypothesis_id": hyp_id})

    def get_by_goal(
        self, goal_text: str, limit: int = 10, source=None
    ) -> list[HypothesisORM]:
        """
        Returns all hypotheses for a given goal.
        """
        query = (
            self.session.query(HypothesisORM)
            .join(GoalORM)
            .filter(GoalORM.goal_text == goal_text)
        )

        if source:
            from co_ai.models import EvaluationORM
            query = query.join(EvaluationORM).filter(EvaluationORM.source == source)

        return query.limit(limit).all()

    def get_latest(self, goal_text: str, limit: int = 10) -> list[HypothesisORM]:
        return self.session.query(HypothesisORM).join(GoalORM).filter(
            GoalORM.goal_text == goal_text
        ).order_by(HypothesisORM.created_at.desc()).limit(limit).all()

    def get_unreflected(self, goal_text: str, limit: int = 10) -> list[HypothesisORM]:
        return self.session.query(HypothesisORM).join(GoalORM).filter(
            GoalORM.goal_text == goal_text,
            HypothesisORM.reflection.is_(None)
        ).limit(limit).all()

    def get_unreviewed(self, goal_text: str, limit: int = 10) -> list[HypothesisORM]:
        return self.session.query(HypothesisORM).join(GoalORM).filter(
            GoalORM.goal_text == goal_text,
            HypothesisORM.review.is_(None)
        ).limit(limit).all()

    def get_from_text(self, query: str, threshold: float = 0.95) -> Optional[HypothesisORM]:
        """
        Finds exact or fuzzy match for hypothesis text.
        """
        result = self.session.query(HypothesisORM).filter(HypothesisORM.text == query).first()
        if result:
            return result

        # Fallback to similarity search if needed
        # This requires pg_trgm extension in PostgreSQL
        result = self.session.query(HypothesisORM).filter(
            HypothesisORM.text.ilike(f"%{query}%")
        ).first()

        if result and result.text:
            sim = SequenceMatcher(None, result.text, query).ratio()
            if sim >= threshold:
                return result

        return None

    def get_by_id(self, hyp_id: int) -> Optional[HypothesisORM]:
        return self.session.get(HypothesisORM, hyp_id)

    def get_all(self, limit: int = 100) -> list[HypothesisORM]:
        return self.session.query(HypothesisORM).order_by(HypothesisORM.created_at.desc()).limit(limit).all()
    
    def get_similar(self, query: str, limit: int = 3) -> list[str]:
        """
        Get top N hypotheses similar to the given prompt using semantic similarity.

        Args:
            query (str): New hypothesis or idea
            limit (int): Number of similar items to return

        Returns:
            list: Top N similar hypotheses
        """
        try:
            query_embedding = self.embedding_store.get_or_create(query)

            results = []
            with self.embedding_store.conn.cursor() as cur:
                cur.execute(
                    "SELECT text FROM hypotheses ORDER BY embedding <-> %s LIMIT %s",
                    (np.array(query_embedding), limit),
                )
                results = [row[0] for row in cur.fetchall()]

            if self.logger:
                self.logger.log("SimilarHypothesesFound", {
                    "query": query[:100],
                    "matches": [r[:100] for r in results]
                })

            return results

        except Exception as e:
            if self.logger:
                self.logger.log("SimilarHypothesesSearchFailed", {"error": str(e)})
            return []