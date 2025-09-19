# stephanie/memory/document_store.py
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import desc

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.document import DocumentORM


class DocumentStore(BaseSQLAlchemyStore):
    orm_model = DocumentORM
    default_order_by = "id"  # use column name string for BaseSQLAlchemyStore

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "documents"

    def name(self) -> str:
        return self.name

    # ---------- Writes ----------

    def add_document(self, doc: dict) -> DocumentORM:
        """Simple insert; returns the persisted ORM row."""
        def op(s):
            
                document = DocumentORM(
                    title=doc["title"],
                    source=doc["source"],
                    external_id=doc.get("external_id"),
                    url=doc.get("url"),
                    text=doc.get("text"),
                    summary=doc.get("summary"),
                    goal_id=doc.get("goal_id"),
                )
                s.add(document)
                s.flush()  # assign id
                return document
        return self._run(op)

    def bulk_add_documents(self, documents: list[dict]) -> list[DocumentORM]:
        def op(s):
            
                orm_docs = [
                    DocumentORM(
                        title=doc["title"],
                        source=doc["source"],
                        external_id=doc.get("external_id"),
                        url=doc.get("url"),
                        text=doc.get("text"),
                        summary=doc.get("summary"),
                        goal_id=doc.get("goal_id"),
                    )
                    for doc in documents
                ]
                s.add_all(orm_docs)
                s.flush()
                return orm_docs
        return self._run(op)

    # ---------- Reads ----------

    def get_by_id(self, document_id: int) -> Optional[DocumentORM]:
        def op(s):
            
                return s.get(DocumentORM, document_id)
        return self._run(op)

    def get_by_url(self, url: str) -> Optional[DocumentORM]:
        def op(s):
            
                return s.query(DocumentORM).filter_by(url=url).first()
        return self._run(op)

    def get_all(self, limit: int = 100) -> List[DocumentORM]:
        def op(s):
            
                return (
                    s.query(DocumentORM)
                    .order_by(
                        desc(getattr(DocumentORM, "created_at", DocumentORM.id))
                    )
                    .limit(limit)
                    .all()
                )
        return self._run(op)

    def get_by_ids(self, document_ids: list[int]) -> List[DocumentORM]:
        def op(s):
            
                return s.query(DocumentORM).filter(DocumentORM.id.in_(document_ids)).all()
        return self._run(op)

    # ---------- Deletes ----------

    def delete_by_id(self, document_id: int) -> bool:
        def op(s):
            
                doc = s.get(DocumentORM, document_id)
                if not doc:
                    return False
                s.delete(doc)
                # commit happens via scope
                return True
        return self._run(op)
