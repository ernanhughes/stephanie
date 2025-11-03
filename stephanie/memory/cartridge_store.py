# stephanie/memory/cartridge_store.py
from __future__ import annotations

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.scorable_embedding import ScorableEmbeddingORM
from stephanie.models.theorem import CartridgeORM


class CartridgeStore(BaseSQLAlchemyStore):
    orm_model = CartridgeORM
    default_order_by = CartridgeORM.id  # use column for BaseSQLAlchemyStore

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "cartridges"

    def _ensure_scorable_embedding(
        self,
        s,
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
            s.query(ScorableEmbeddingORM)
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
        s.add(new_se)
        s.flush()  # populate new_se.id without commit
        return new_se.id

    def add_cartridge(self, data: dict) -> CartridgeORM:
        def op(s):
            existing = (
                s.query(CartridgeORM)
                .filter_by(
                    source_type=data["source_type"],
                    source_uri=data["source_uri"],
                )
                .first()
            )

            if data.get("embedding_id"):
                resolved_id = self._ensure_scorable_embedding(
                    s,
                    embedding_id=data["embedding_id"],
                    scorable_id=data.get("source_uri"),
                    scorable_type=data["source_type"],
                    embedding_type=data.get("embedding_type", "unknown"),
                )
                data["embedding_id"] = resolved_id

            if existing:
                for field in [
                    "markdown_content",
                    "title",
                    "summary",
                    "sections",
                    "triples",
                    "domain_tags",
                    "embedding_id",
                ]:
                    if field in data:
                        setattr(existing, field, data[field])
                return existing

            cartridge = CartridgeORM(**data)
            s.add(cartridge)
            return cartridge

        return self._run(op)

    def bulk_add_cartridges(self, items: list[dict]) -> list[CartridgeORM]:
        def op(s):
            cartridges = []
            for item in items:
                if item.get("embedding_id"):
                    resolved_id = self._ensure_scorable_embedding(
                        s,
                        embedding_id=item["embedding_id"],
                        scorable_id=item.get("source_uri"),
                        scorable_type=item["source_type"],
                        embedding_type=item.get("embedding_type", "unknown"),
                    )
                    item["embedding_id"] = resolved_id
                cartridges.append(CartridgeORM(**item))

            s.add_all(cartridges)
            return cartridges

        return self._run(op)

    def get_by_id(self, cartridge_id: int) -> CartridgeORM | None:
        def op(s):
            return s.query(CartridgeORM).filter_by(id=cartridge_id).first()

        return self._run(op)

    def get_by_source_uri(
        self, uri: str, source_type: str
    ) -> CartridgeORM | None:
        def op(s):
            return (
                s.query(CartridgeORM)
                .filter_by(source_uri=uri, source_type=source_type)
                .first()
            )

        return self._run(op)

    def get_all(self, limit=100) -> list[CartridgeORM]:
        def op(s):
            return s.query(CartridgeORM).limit(limit).all()

        return self._run(op)

    def delete_by_id(self, cartridge_id: int) -> bool:
        def op(s):
            cartridge = (
                s.query(CartridgeORM).filter_by(id=cartridge_id).first()
            )
            if cartridge:
                s.delete(cartridge)
                return True
            return False

        return self._run(op)

    def get_run_id(self, pipeline_run_id: int) -> list[CartridgeORM]:
        def op(s):
            return (
                s.query(CartridgeORM)
                .filter_by(pipeline_run_id=pipeline_run_id)
                .all()
            )

        return self._run(op)
