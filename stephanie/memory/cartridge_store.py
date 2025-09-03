# stephanie/memory/cartridge_store.py
from sqlalchemy.orm import Session

from stephanie.models.scorable_embedding import ScorableEmbeddingORM
from stephanie.models.theorem import CartridgeORM


class CartridgeStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "cartridges"

    def _ensure_scorable_embedding(
        self,
        embedding_id: int,
        scorable_id: str,
        scorable_type: str,
        embedding_type: str,
    ) -> int:
        """
        Ensure a corresponding entry exists in scorable_embeddings.
        Returns scorable_embeddings.id to store on CartridgeORM/TheoremORM as a soft pointer.
        """
        existing = (
            self.session.query(ScorableEmbeddingORM)
            .filter_by(
                scorable_id=scorable_id,
                scorable_type=scorable_type,
                embedding_id=embedding_id,
                embedding_type=embedding_type,
            )
            .first()
        )
        if existing:
            return existing.id

        new_se = ScorableEmbeddingORM(
            scorable_id=scorable_id,
            scorable_type=scorable_type,
            embedding_id=embedding_id,
            embedding_type=embedding_type,
        )
        self.session.add(new_se)
        self.session.flush()  # populate new_se.id without commit
        return new_se.id

    def add_cartridge(self, data: dict) -> CartridgeORM:
        existing = (
            self.session.query(CartridgeORM)
            .filter_by(source_type=data["source_type"], source_uri=data["source_uri"])
            .first()
        )

        # Resolve external embedding pointer into scorable_embeddings.id
        if data.get("embedding_id"):
            resolved_id = self._ensure_scorable_embedding(
                embedding_id=data["embedding_id"],
                scorable_id=data.get("source_uri"),   # the owner key (e.g., doc uri)
                scorable_type=data["source_type"],    # 'document', 'hypothesis', etc.
                embedding_type=data.get("embedding_type", "unknown"),
            )
            data["embedding_id"] = resolved_id  # store soft pointer id

        if existing:
            for field in [
                "markdown_content", "title", "summary",
                "sections", "triples", "domain_tags", "embedding_id"
            ]:
                if field in data:
                    setattr(existing, field, data[field])
            self.session.commit()
            return existing

        cartridge = CartridgeORM(**data)
        self.session.add(cartridge)
        self.session.commit()
        return cartridge

    def bulk_add_cartridges(self, items: list[dict]) -> list[CartridgeORM]:
        cartridges = []
        for item in items:
            if item.get("embedding_id"):
                resolved_id = self._ensure_scorable_embedding(
                    embedding_id=item["embedding_id"],
                    scorable_id=item.get("source_uri"),
                    scorable_type=item["source_type"],
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

    def get_run_id(self, pipeline_run_id: int) -> list[CartridgeORM]:
        return (
            self.session.query(CartridgeORM)
            .filter_by(pipeline_run_id=pipeline_run_id)
            .all()
        )   