# stephanie/memory/document_store.py

from sqlalchemy.orm import Session

from stephanie.models.document import DocumentORM


class DocumentStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "document"

    def add_document(self, doc: dict) -> DocumentORM:
        document = DocumentORM(
            title=doc["title"],
            source=doc["source"],
            external_id=doc.get("external_id"),
            url=doc.get("url"),
            content=doc.get("text") or doc.get("content"),  # support both fields
            summary=doc.get("summary"),
            goal_id=doc.get("goal_id"),
        )
        self.session.add(document)
        self.session.commit()
        return document

    def bulk_add_documents(self, documents: list[dict]) -> list[DocumentORM]:
        orm_docs = [
            DocumentORM(
                title=doc["title"],
                source=doc["source"],
                external_id=doc.get("external_id"),
                url=doc.get("url"),
                content=doc.get("content"),
            )
            for doc in documents
        ]
        self.session.add_all(orm_docs)
        self.session.commit()
        return orm_docs

    def get_by_id(self, document_id: int) -> DocumentORM | None:
        return self.session.query(DocumentORM).filter_by(id=document_id).first()

    def get_by_url(self, url: str) -> DocumentORM | None:
        return self.session.query(DocumentORM).filter_by(url=url).first()

    def get_all(self, limit=100) -> list[DocumentORM]:
        return self.session.query(DocumentORM).limit(limit).all()

    def delete_by_id(self, document_id: int) -> bool:
        doc = self.get_by_id(document_id)
        if doc:
            self.session.delete(doc)
            self.session.commit()
            return True
        return False
