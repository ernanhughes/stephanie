# stephanie/memory/theorem_store.py
from __future__ import annotations

from typing import Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.theorem import TheoremORM


class TheoremStore(BaseSQLAlchemyStore):
    orm_model = TheoremORM
    default_order_by = TheoremORM.created_at

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "theorems"

    def name(self) -> str:
        return self.name

    def add_theorem(self, data: dict) -> TheoremORM:
        """
        Adds a new theorem or updates an existing one if a matching statement already exists.
        """
        existing = (
            self.session.query(TheoremORM)
            .filter_by(statement=data["statement"])
            .first()
        )

        if existing:
            existing.proof = data.get("proof", existing.proof)
            existing.embedding_id = data.get(
                "embedding_id", existing.embedding_id
            )
            self.session.commit()
            return existing

        theorem = TheoremORM(**data)
        self.session.add(theorem)
        self.session.commit()
        return theorem

    def bulk_add_theorems(self, items: list[dict]) -> list[TheoremORM]:
        theorems = [
            TheoremORM(
                statement=item["statement"],
                proof=item.get("proof"),
                embedding_id=item.get("embedding_id"),
            )
            for item in items
        ]
        self.session.add_all(theorems)
        self.session.commit()
        return theorems

    def get_by_id(self, theorem_id: int) -> Optional[TheoremORM]:
        return self.session.query(TheoremORM).filter_by(id=theorem_id).first()

    def get_by_statement(self, statement: str) -> Optional[TheoremORM]:
        return (
            self.session.query(TheoremORM)
            .filter_by(statement=statement)
            .first()
        )

    def get_all(self, limit=100) -> list[TheoremORM]:
        return self.session.query(TheoremORM).limit(limit).all()

    def delete_by_id(self, theorem_id: int) -> bool:
        theorem = self.get_by_id(theorem_id)
        if theorem:
            self.session.delete(theorem)
            self.session.commit()
            return True
        return False

    def get_by_run_id(self, run_id: int) -> list[TheoremORM]:
        return (
            self.session.query(TheoremORM)
            .filter_by(pipeline_run_id=run_id)
            .all()
        )
