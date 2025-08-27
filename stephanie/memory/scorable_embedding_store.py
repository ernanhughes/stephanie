from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stephanie.models.scorable_embedding import ScorableEmbeddingORM


class ScorableEmbeddingStore:
    """
    Store for embeddings linked to any Scorable
    (documents, plan_traces, prompts, responses, etc.).
    """

    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "scorable_embeddings"

    def insert(self, data: dict) -> int:
        """
        Insert a new embedding record.

        Expected keys:
            - scorable_id (str)
            - scorable_type (str)
            - embedding_id (int)
            - embedding_type (str)
        """
        obj = ScorableEmbeddingORM(**data, created_at=datetime.now())
        self.session.add(obj)
        self.session.commit()

        if self.logger:
            self.logger.log("ScorableEmbeddingInserted", data)

        return obj.id

    def get_by_scorable(
        self, scorable_id: str, scorable_type: str, embedding_type: str | None = None
    ) -> list[ScorableEmbeddingORM]:
        """
        Fetch all embeddings for a given scorable (optionally filtered by embedding type).
        """
        q = self.session.query(ScorableEmbeddingORM).filter_by(
            scorable_id=str(scorable_id), scorable_type=scorable_type
        )
        if embedding_type:
            q = q.filter_by(embedding_type=embedding_type)
        return q.all()

    def get_embedding_id(
        self, scorable_id: str, scorable_type: str, embedding_type: str
    ) -> int | None:
        """
        Get a specific embedding_id for a scorable, if it exists.
        """
        rec = (
            self.session.query(ScorableEmbeddingORM)
            .filter_by(
                scorable_id=str(scorable_id),
                scorable_type=scorable_type,
                embedding_type=embedding_type,
            )
            .first()
        )
        return rec.embedding_id if rec else None

    def get_or_create(
        self,
        scorable_id: str,
        scorable_type: str,
        embedding_id: int,
        embedding_type: str,
    ) -> int:
        """
        Return existing row or create a new one safely.
        Ensures uniqueness on (scorable_id, scorable_type, embedding_type).
        """
        existing = (
            self.session.query(ScorableEmbeddingORM)
            .filter_by(
                scorable_id=str(scorable_id),
                scorable_type=scorable_type,
                embedding_type=embedding_type,
            )
            .first()
        )
        if existing:
            return existing.id

        obj = ScorableEmbeddingORM(
            scorable_id=str(scorable_id),
            scorable_type=scorable_type,
            embedding_id=embedding_id,
            embedding_type=embedding_type,
            created_at=datetime.now(),
        )
        self.session.add(obj)
        try:
            self.session.commit()
            if self.logger:
                self.logger.log(
                    "ScorableEmbeddingInserted",
                    {
                        "scorable_id": str(scorable_id),
                        "scorable_type": scorable_type,
                        "embedding_type": embedding_type,
                        "embedding_id": embedding_id,
                    },
                )
            return obj.id
        except IntegrityError:
            self.session.rollback()
            # Another transaction inserted it first → fetch again
            existing = (
                self.session.query(ScorableEmbeddingORM)
                .filter_by(
                    scorable_id=str(scorable_id),
                    scorable_type=scorable_type,
                    embedding_type=embedding_type,
                )
                .first()
            )
            if existing:
                return existing.id
            raise
