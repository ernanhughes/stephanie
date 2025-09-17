# stephanie/memory/cartridge_domain_store.py
from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.cartridge_domain import CartridgeDomainORM


class CartridgeDomainStore(BaseSQLAlchemyStore):
    orm_model = CartridgeDomainORM
    default_order_by = CartridgeDomainORM.id.desc()
    
    def __init__(self, session: Session, logger=None):
        super().__init__(session, logger)
        self.name = "cartridge_domains"

    def name(self) -> str:
        return self.name

    def insert(self, data: dict) -> CartridgeDomainORM:
        """
        Insert or update a domain classification entry manually.

        Expected dict keys: cartridge_id, domain, score
        """
        # Try to find existing entry
        existing = (
            self.session.query(CartridgeDomainORM)
            .filter_by(cartridge_id=data["cartridge_id"], domain=data["domain"])
            .first()
        )

        if existing:
            # Update score if it has changed
            if existing.score != data["score"]:
                existing.score = data["score"]
                self.session.commit()
                if self.logger:
                    self.logger.log("CartridgeDomainUpdated", data)
            return existing
        else:
            # Create new entry
            domain_obj = CartridgeDomainORM(**data)
            self.session.add(domain_obj)
            self.session.commit()
            if self.logger:
                self.logger.log("CartridgeDomainInserted", data)
            return domain_obj

    def get_domains(self, cartridge_id: int) -> list[CartridgeDomainORM]:
        return (
            self.session.query(CartridgeDomainORM)
            .filter_by(cartridge_id=cartridge_id)
            .order_by(CartridgeDomainORM.score.desc())
            .all()
        )

    def delete_domains(self, cartridge_id: int):
        self.session.query(CartridgeDomainORM).filter_by(
            cartridge_id=cartridge_id
        ).delete()
        self.session.commit()
        if self.logger:
            self.logger.log("CartridgeDomainsDeleted", {"cartridge_id": cartridge_id})

    def set_domains(self, cartridge_id: int, domains: list[tuple[str, float]]):
        """
        Clear and re-add domains for the cartridge.
        :param domains: list of (domain, score) pairs
        """
        self.delete_domains(cartridge_id)
        for domain, score in domains:
            self.insert(
                {
                    "cartridge_id": cartridge_id,
                    "domain": domain,
                    "score": float(score),
                }
            )
