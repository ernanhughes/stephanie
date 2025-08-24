# stephanie/memory/document_embedding_store.py

from sqlalchemy.orm import Session

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

    def get_by_document(self, document_id: str, document_type: str, embedding_type: str = None):
        q = (
            self.session.query(DocumentEmbeddingORM)
            .filter_by(document_id=document_id, document_type=document_type)
        )
        if embedding_type:
            q = q.filter_by(embedding_type=embedding_type)
        return q.all()

    def get_embedding_id(self, document_id: str, document_type: str, embedding_type: str) -> int | None:
        rec = (
            self.session.query(DocumentEmbeddingORM)
            .filter_by(document_id=document_id, document_type=document_type, embedding_type=embedding_type)
            .first()
        )
        return rec.embedding_id if rec else None
