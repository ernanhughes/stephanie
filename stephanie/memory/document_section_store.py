# stephanie/memory/document_section_store.py
from __future__ import annotations

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.document_section import DocumentSectionORM


class DocumentSectionStore(BaseSQLAlchemyStore):
    orm_model = DocumentSectionORM
    default_order_by = "id"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "document_sections"

    def name(self) -> str:
        return self.name

    def insert(self, section_dict: dict) -> DocumentSectionORM:
        def op(s):
            section = DocumentSectionORM(**section_dict)
            s.add(section)
            s.flush()
            if self.logger:
                self.logger.log("SectionInserted", section.to_dict())
            return section

        return self._run(op)

    def upsert(self, section_dict: dict) -> DocumentSectionORM:
        """
        Update or insert a document section based on document_id and section_name.
        """

        def op(s):
            existing = (
                s.query(DocumentSectionORM)
                .filter_by(
                    document_id=section_dict["document_id"],
                    section_name=section_dict["section_name"],
                )
                .first()
            )

            if existing:
                for key, value in section_dict.items():
                    setattr(existing, key, value)
                action = "SectionUpdated"
            else:
                existing = DocumentSectionORM(**section_dict)
                s.add(existing)
                action = "SectionInserted"

            if self.logger:
                self.logger.log(action, existing.to_dict())
            return existing

        return self._run(op)

    def get_by_document(self, document_id: int) -> list[DocumentSectionORM]:
        def op(s):
            return (
                s.query(DocumentSectionORM)
                .filter_by(document_id=document_id)
                .order_by(DocumentSectionORM.id)
                .all()
            )

        return self._run(op)

    def delete_by_document(self, document_id: int) -> None:
        def op(s):
            s.query(DocumentSectionORM).filter_by(
                document_id=document_id
            ).delete()

        return self._run(op)
