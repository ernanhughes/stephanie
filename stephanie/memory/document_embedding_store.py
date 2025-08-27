# stephanie/memory/document_embedding_store.py

from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from stephanie.models.document_embedding import DocumentEmbeddingORM


class DocumentEmbeddingStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "document_embeddings"

    def insert(self, data: dict) -> int:
        obj = DocumentEmbeddingORM(**data)
        self.session.add(obj)
        self.session.commit()
        if self.logger:
            self.logger.log("DocumentEmbeddingInserted", data)
        return obj.id

    def get_by_document(
        self, document_id: str, document_type: str, embedding_type: str = None
    ):
        q = self.session.query(DocumentEmbeddingORM).filter_by(
            document_id=str(document_id), document_type=document_type
        )
        if embedding_type:
            q = q.filter_by(embedding_type=embedding_type)
        return q.all()

    def get_embedding_id(
        self, document_id: str, document_type: str, embedding_type: str
    ) -> int | None:
        rec = (
            self.session.query(DocumentEmbeddingORM)
            .filter_by(
                document_id=str(document_id),
                document_type=document_type,
                embedding_type=embedding_type,
            )
            .first()
        )
        return rec.embedding_id if rec else None

    def get_or_create(
        self,
        document_id: str,
        document_type: str,
        embedding_id: int,
        embedding_type: str,
    ) -> int:
        """Return existing row or create a new one safely."""
        existing = (
            self.session.query(DocumentEmbeddingORM)
            .filter_by(
                document_id=str(document_id),
                document_type=document_type,
                embedding_type=embedding_type,
            )
            .first()
        )
        if existing:
            return existing.id

        obj = DocumentEmbeddingORM(
            document_id=str(document_id),
            document_type=document_type,
            embedding_id=embedding_id,
            embedding_type=embedding_type,
            created_at=datetime.now(),
        )
        self.session.add(obj)
        try:
            self.session.commit()
            if self.logger:
                self.logger.log(
                    "DocumentEmbeddingInserted",
                    {
                        "document_id": str(document_id),
                        "document_type": document_type,
                        "embedding_type": embedding_type,
                        "embedding_id": embedding_id,
                    },
                )
            return obj.id
        except IntegrityError:
            self.session.rollback()
            # Another transaction inserted it first → fetch again
            existing = (
                self.session.query(DocumentEmbeddingORM)
                .filter_by(
                    document_id=str(document_id),
                    document_type=document_type,
                    embedding_type=embedding_type,
                )
                .first()
            )
            if existing:
                return existing.id
            raise
