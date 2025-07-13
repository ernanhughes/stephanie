# stephanie/memory/document_domain_store.py
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from stephanie.models.document_domain import DocumentDomainORM


class DocumentDomainStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "document_domains"

    def insert(self, data: dict) -> DocumentDomainORM:
        """
        Insert or update a domain classification entry.

        Expected dict keys: document_id, domain, score
        """
        stmt = (
            pg_insert(DocumentDomainORM)
            .values(**data)
            .on_conflict_do_nothing(index_elements=["document_id", "domain"])
            .returning(DocumentDomainORM.id)  # Adjust field if needed
        )

        result = self.session.execute(stmt)
        inserted_id = result.scalar()
        self.session.commit()

        if inserted_id:
            self.logger.log("DomainUpserted", data)

        # Optionally return the upserted object (retrieved fresh)
        return (
            self.session.query(DocumentDomainORM)
            .filter_by(document_id=data["document_id"], domain=data["domain"])
            .first()
        )

    def get_domains(self, document_id: int) -> list[DocumentDomainORM]:
        return (
            self.session.query(DocumentDomainORM)
            .filter_by(document_id=document_id)
            .order_by(DocumentDomainORM.score.desc())
            .all()
        )

    def delete_domains(self, document_id: int):
        self.session.query(DocumentDomainORM).filter_by(
            document_id=document_id
        ).delete()
        self.session.commit()
        if self.logger:
            self.logger.log("DomainsDeleted", {"document_id": document_id})

    def set_domains(self, document_id: int, domains: list[tuple[str, float]]):
        """
        Clear and re-add domains for the document.
        :param domains: list of (domain, score) pairs
        """
        self.delete_domains(document_id)
        for domain, score in domains:
            self.insert(
                {
                    "document_id": document_id,
                    "domain": domain,
                    "score": float(score),
                }
            )
