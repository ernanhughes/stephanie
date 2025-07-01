from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from stephanie.models.document_section_domain import DocumentSectionDomainORM


class DocumentSectionDomainStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "document_section_domains"

    def insert(self, data: dict) -> DocumentSectionDomainORM:
        """
        Insert or update a domain classification entry for a document section.

        Expected keys: document_section_id, domain, score
        """
        stmt = (
            pg_insert(DocumentSectionDomainORM)
            .values(**data)
            .on_conflict_do_nothing(index_elements=["document_section_id", "domain"])
            .returning(DocumentSectionDomainORM.document_section_id)
        )

        result = self.session.execute(stmt)
        inserted_id = result.scalar()
        self.session.commit()

        if inserted_id:
            self.logger.log("SectionDomainInserted", data)

        return (
            self.session.query(DocumentSectionDomainORM)
            .filter_by(document_section_id=data["document_section_id"], domain=data["domain"])
            .first()
        )

    def get_domains(self, document_section_id: int) -> list[DocumentSectionDomainORM]:
        return (
            self.session.query(DocumentSectionDomainORM)
            .filter_by(document_section_id=document_section_id)
            .order_by(DocumentSectionDomainORM.score.desc())
            .all()
        )

    def delete_domains(self, document_section_id: int):
        self.session.query(DocumentSectionDomainORM).filter_by(document_section_id=document_section_id).delete()
        self.session.commit()

        if self.logger:
            self.logger.log("SectionDomainsDeleted", {"document_section_id": document_section_id})

    def set_domains(self, document_section_id: int, domains: list[tuple[str, float]]):
        """
        Clear and re-add domains for the document section.

        :param domains: list of (domain, score) tuples
        """
        self.delete_domains(document_section_id)
        for domain, score in domains:
            self.insert({
                "document_section_id": document_section_id,
                "domain": domain,
                "score": float(score),
            })
