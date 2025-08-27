# stephanie/memory/cartridge_store.py

from sqlalchemy.orm import Session
from stephanie.models.theorem import CartridgeORM
from stephanie.models.document_embedding import DocumentEmbeddingORM


class CartridgeStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "cartridges"

    def _ensure_document_embedding(self, embedding_id: int, document_id: str, document_type: str, embedding_type: str) -> int:
        """
        Ensure a corresponding entry exists in document_embeddings.
        Returns the document_embeddings.id to use in CartridgeORM.
        """
        existing = (
            self.session.query(DocumentEmbeddingORM)
            .filter_by(
                document_id=document_id,
                document_type=document_type,
                embedding_id=embedding_id,
                embedding_type=embedding_type,
            )
            .first()
        )
        if existing:
            return existing.id

        new_de = DocumentEmbeddingORM(
            document_id=document_id,
            document_type=document_type,
            embedding_id=embedding_id,
            embedding_type=embedding_type,
        )
        self.session.add(new_de)
        self.session.flush()  # ensures new_de.id is populated without commit
        return new_de.id

    def add_cartridge(self, data: dict) -> CartridgeORM:
        existing = (
            self.session.query(CartridgeORM)
            .filter_by(source_type=data["source_type"], source_uri=data["source_uri"])
            .first()
        )

        # Resolve embedding_id into document_embeddings
        if data.get("embedding_id"):
            resolved_id = self._ensure_document_embedding(
                embedding_id=data["embedding_id"],
                document_id=data.get("source_uri"),   # maps back to document/hypothesis/etc.
                document_type=data["source_type"],
                embedding_type=data.get("embedding_type", "unknown"),
            )
            data["embedding_id"] = resolved_id

        if existing:
            # Update existing cartridge
            for field in [
                "markdown_content", "title", "summary",
                "sections", "triples", "domain_tags", "embedding_id"
            ]:
                if field in data:
                    setattr(existing, field, data[field])
            self.session.commit()
            return existing

        # Insert new
        cartridge = CartridgeORM(**data)
        self.session.add(cartridge)
        self.session.commit()
        return cartridge

    def bulk_add_cartridges(self, items: list[dict]) -> list[CartridgeORM]:
        cartridges = []
        for item in items:
            if item.get("embedding_id"):
                resolved_id = self._ensure_document_embedding(
                    embedding_id=item["embedding_id"],
                    document_id=item.get("source_uri"),
                    document_type=item["source_type"],
                    embedding_type=item.get("embedding_type", "unknown"),
                )
                item["embedding_id"] = resolved_id
            cartridges.append(CartridgeORM(**item))

        self.session.add_all(cartridges)
        self.session.commit()
        return cartridges

    def get_by_id(self, cartridge_id: int) -> CartridgeORM | None:
        return self.session.query(CartridgeORM).filter_by(id=cartridge_id).first()

    def get_by_source_uri(self, uri: str, source_type: str) -> CartridgeORM | None:
        return (
            self.session.query(CartridgeORM)
            .filter_by(source_uri=uri, source_type=source_type)
            .first()
        )

    def get_all(self, limit=100) -> list[CartridgeORM]:
        return self.session.query(CartridgeORM).limit(limit).all()

    def delete_by_id(self, cartridge_id: int) -> bool:
        cartridge = self.get_by_id(cartridge_id)
        if cartridge:
            self.session.delete(cartridge)
            self.session.commit()
            return True
        return False
