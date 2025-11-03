# stephanie/memory/document_domain_section_store.py
from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert as pg_insert

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.document_section_domain import DocumentSectionDomainORM


class DocumentSectionDomainStore(BaseSQLAlchemyStore):
    orm_model = DocumentSectionDomainORM
    default_order_by = "id"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "document_section_domains"

    def insert(self, data: dict) -> DocumentSectionDomainORM:
        """
        Insert or update a domain classification entry for a document section.

        Expected keys: document_section_id, domain, score
        """
        def op(s):
            
                stmt = (
                    pg_insert(DocumentSectionDomainORM)
                    .values(**data)
                    .on_conflict_do_nothing(
                        index_elements=["document_section_id", "domain"]
                    )
                    .returning(DocumentSectionDomainORM.document_section_id)
                )
                result = s.execute(stmt)
                inserted_id = result.scalar()
                if inserted_id and self.logger:
                    self.logger.log("SectionDomainInserted", data)

                return (
                    s.query(DocumentSectionDomainORM)
                    .filter_by(
                        document_section_id=data["document_section_id"],
                        domain=data["domain"],
                    )
                    .first()
                )
        return self._run(op)

    def get_domains(self, document_section_id: int) -> list[DocumentSectionDomainORM]:
        return self._run(
            lambda: (
                self._scope()
                .query(DocumentSectionDomainORM)
                .filter_by(document_section_id=document_section_id)
                .order_by(DocumentSectionDomainORM.score.desc())
                .all()
            ),
            default=[],
        )

    def delete_domains(self, document_section_id: int):
        def op(s):
            
                s.query(DocumentSectionDomainORM).filter_by(
                    document_section_id=document_section_id
                ).delete()
                if self.logger:
                    self.logger.log(
                        "SectionDomainsDeleted",
                        {"document_section_id": document_section_id},
                    )
        return self._run(op)

    def set_domains(self, document_section_id: int, domains: list[tuple[str, float]]):
        """
        Clear and re-add domains for the document section.

        :param domains: list of (domain, score) tuples
        """
        def op(s):
            self.delete_domains(document_section_id)
            for domain, score in domains:
                self.insert(
                    {
                        "document_section_id": document_section_id,
                        "domain": domain,
                        "score": float(score),
                    }
                )
        return self._run(op)
