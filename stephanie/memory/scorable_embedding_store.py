# stephanie/memory/scorable_embedding_store.py
from __future__ import annotations

from datetime import datetime

from sqlalchemy.exc import IntegrityError

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.scorable_embedding import ScorableEmbeddingORM
from stephanie.scoring.scorable import Scorable


class ScorableEmbeddingStore(BaseSQLAlchemyStore):
    """
    Store for embeddings linked to any Scorable
    (documents, plan_traces, prompts, responses, etc.).
    """
    orm_model = ScorableEmbeddingORM
    default_order_by = ScorableEmbeddingORM.created_at.asc()

    def __init__(self, session_or_maker, logger=None, embedding=None):
        super().__init__(session_or_maker, logger)
        self.name = "scorable_embeddings"
        self.embedding = embedding

    def insert(self, data: dict) -> int:
        """
        Insert a new embedding record.

        Expected keys:
            - scorable_id (str)
            - scorable_type (str)
            - embedding_id (int)
            - embedding_type (str)
        """
        def op(s):
            obj = ScorableEmbeddingORM(**data, created_at=datetime.now())
            
            s.add(obj)
            if self.logger:
                self.logger.log("ScorableEmbeddingInserted", data)
            return obj.id
        return self._run(op)

    def insert_scorable(self, scorable: Scorable) -> int:
        """
        Insert a new embedding record for a Scorable object.
        """
        def op(s):
            embedding_id = self.embedding.get_id_for_text(scorable.text)
            data = {
                "scorable_id": str(scorable.id),
                "scorable_type": scorable.target_type,
                "embedding_id": embedding_id,
                "embedding_type": self.embedding.name,
            }
            obj = ScorableEmbeddingORM(**data, created_at=datetime.now())
            
            s.add(obj)
            if self.logger:
                self.logger.log("ScorableEmbeddingInserted", data)
            return obj.id
        return self._run(op)

    def get_by_scorable(
        self, scorable_id: str, scorable_type: str, embedding_type: str | None = None
    ) -> list[ScorableEmbeddingORM]:
        """Fetch all embeddings for a given scorable (optionally filtered by embedding type)."""
        def op(s):
            q = s.query(ScorableEmbeddingORM).filter_by(
                scorable_id=str(scorable_id), scorable_type=scorable_type
            )
            if embedding_type:
                q = q.filter_by(embedding_type=embedding_type)
            return q.all()
        return self._run(op)

    def get_embedding_id(
        self, scorable_id: str, scorable_type: str, embedding_type: str
    ) -> int | None:
        """Get a specific embedding_id for a scorable, if it exists."""
        def op(s):
            rec = (
                s.query(ScorableEmbeddingORM)
                .filter_by(
                    scorable_id=str(scorable_id),
                    scorable_type=scorable_type,
                    embedding_type=embedding_type,
                )
                .first()
            )
            return rec.embedding_id if rec else None
        return self._run(op)

    def get_or_create(self, scorable: Scorable) -> int:
        """
        Return existing row or create a new one safely.
        Ensures uniqueness on (scorable_id, scorable_type, embedding_type).
        """
        def op(s):
            
            existing = (
                s.query(ScorableEmbeddingORM)
                .filter_by(
                    scorable_id=str(scorable.id),
                    scorable_type=scorable.target_type,
                    embedding_type=self.embedding.name,
                )
                .first()
            )
            if existing:
                return existing.id

            try:
                # Generate embedding if needed
                self.embedding.get_or_create(scorable.text)
                embedding_id = self.embedding.get_id_for_text(scorable.text)
                data = {
                    "scorable_id": str(scorable.id),
                    "scorable_type": scorable.target_type,
                    "embedding_id": embedding_id,
                    "embedding_type": self.embedding.name,
                }
                obj = ScorableEmbeddingORM(**data, created_at=datetime.now())
                s.add(obj)
                if self.logger:
                    self.logger.log("ScorableEmbeddingInserted", data)
                return obj.id

            except IntegrityError:
                s.rollback()
                # Another transaction inserted it first → fetch again
                existing = (
                    s.query(ScorableEmbeddingORM)
                    .filter_by(
                        scorable_id=str(scorable.id),
                        scorable_type=scorable.target_type,
                        embedding_type=self.embedding.name,
                    )
                    .first()
                )
                return existing.id if existing else None
        return self._run(op)
