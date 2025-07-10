# stephanie/memory/cartridge_store.py

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from stephanie.models.theorem import CartridgeORM


class CartridgeStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "cartridges"

    def add_cartridge(self, data: dict) -> CartridgeORM:

        existing = (
            self.session.query(CartridgeORM)
            .filter_by(source_type=data["source_type"], source_uri=data["source_uri"])
            .first()
        )

        if existing:
            # Optionally update content
            existing.markdown_content = data.get("markdown_content", existing.markdown_content)
            existing.title = data.get("title", existing.title)
            existing.summary = data.get("summary", existing.summary)
            existing.sections = data.get("sections", existing.sections)
            existing.triples = data.get("triples", existing.triples)
            existing.domain_tags = data.get("domain_tags", existing.domain_tags)
            existing.embedding_id = data.get("embedding_id", existing.embedding_id)
            self.session.commit()
            return existing

        cartridge = CartridgeORM(**data)
        self.session.add(cartridge)
        self.session.commit()
        return cartridge

    def bulk_add_cartridges(self, items: list[dict]) -> list[CartridgeORM]:
        cartridges = [
            CartridgeORM(
                goal_id=item.get("goal_id"),
                source_type=item["source_type"],
                source_uri=item.get("source_uri"),
                markdown_content=item["markdown_content"],
                embedding_id=item.get("embedding_id"),
                title=item.get("title"),
                summary=item.get("summary"),
                sections=item.get("sections"),
                triples=item.get("triples"),
                domain_tags=item.get("domain_tags"),
            )
            for item in items
        ]
        self.session.add_all(cartridges)
        self.session.commit()
        return cartridges

    def get_by_id(self, cartridge_id: int) -> CartridgeORM | None:
        return self.session.query(CartridgeORM).filter_by(id=cartridge_id).first()

    def get_by_source_uri(self, uri: str, source_type: str) -> CartridgeORM | None:
        return self.session.query(CartridgeORM).filter_by(source_uri=uri, source_type=source_type).first()

    def get_all(self, limit=100) -> list[CartridgeORM]:
        return self.session.query(CartridgeORM).limit(limit).all()

    def delete_by_id(self, cartridge_id: int) -> bool:
        cartridge = self.get_by_id(cartridge_id)
        if cartridge:
            self.session.delete(cartridge)
            self.session.commit()
            return True
        return False
