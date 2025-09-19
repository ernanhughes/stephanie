from __future__ import annotations

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.cartridge_domain import CartridgeDomainORM


class CartridgeDomainStore(BaseSQLAlchemyStore):
    orm_model = CartridgeDomainORM
    default_order_by = "id"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "cartridge_domains"

    def insert(self, data: dict) -> CartridgeDomainORM:
        """
        Insert or update a domain classification entry manually.

        Expected dict keys: cartridge_id, domain, score
        """
        def op():
            with self._scope() as s:
                existing = (
                    s.query(CartridgeDomainORM)
                    .filter_by(cartridge_id=data["cartridge_id"], domain=data["domain"])
                    .first()
                )

                if existing:
                    if existing.score != data["score"]:
                        existing.score = data["score"]
                        if self.logger:
                            self.logger.log("CartridgeDomainUpdated", data)
                    return existing
                else:
                    domain_obj = CartridgeDomainORM(**data)
                    s.add(domain_obj)
                    if self.logger:
                        self.logger.log("CartridgeDomainInserted", data)
                    return domain_obj

        return self._run(op)

    def get_domains(self, cartridge_id: int) -> list[CartridgeDomainORM]:
        def op():
            with self._scope() as s:
                return (
                    s.query(CartridgeDomainORM)
                    .filter_by(cartridge_id=cartridge_id)
                    .order_by(CartridgeDomainORM.score.desc())
                    .all()
                )
        return self._run(op)

    def delete_domains(self, cartridge_id: int) -> None:
        def op():
            with self._scope() as s:
                s.query(CartridgeDomainORM).filter_by(cartridge_id=cartridge_id).delete()
                if self.logger:
                    self.logger.log("CartridgeDomainsDeleted", {"cartridge_id": cartridge_id})
        return self._run(op)

    def set_domains(self, cartridge_id: int, domains: list[tuple[str, float]]) -> None:
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
