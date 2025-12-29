# stephanie/memory/theorem_store.py
from __future__ import annotations

from typing import List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.scorable_embedding import ScorableEmbeddingORM
from stephanie.orm.theorem import TheoremORM


class TheoremStore(BaseSQLAlchemyStore):
    orm_model = TheoremORM
    # You can also set to "id" if your Base handles strings; using column keeps it explicit
    default_order_by = TheoremORM.id

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "theorems"


    def _ensure_scorable_embedding(
        self,
        s,
        *,
        embedding_id: int,
        scorable_id: str,
        scorable_type: str = "theorem",
        embedding_type: str = "unknown",
    ) -> int:
        """
        Ensure a row exists in scorable_embeddings tying a 'theorem' to a raw embedding.
        Returns ScorableEmbeddingORM.id (soft pointer to store on TheoremORM.embedding_id).
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
        s.flush()
        return new_se.id

    def _resolve_embedding_field(self, s, data: dict) -> None:
        """
        If 'embedding_id' provided, rewrite it to point to scorable_embeddings.id.
        We use the theorem's unique 'statement' as scorable_id (stable before DB id exists).
        """
        if data.get("embedding_id") is not None:
            resolved_id = self._ensure_scorable_embedding(
                s,
                embedding_id=data["embedding_id"],
                scorable_id=data.get("statement", "unknown-statement"),
                scorable_type="theorem",
                embedding_type=data.get("embedding_type", "unknown"),
            )
            data["embedding_id"] = resolved_id

    # ---------- CRUD ----------

    def insert(self, data: dict) -> TheoremORM:
        def op(s):
            self._resolve_embedding_field(s, data)
            theorem = TheoremORM(**data)
            s.add(theorem)
            s.flush()
            if self.logger:
                self.logger.log("TheoremInserted", theorem.to_dict())
            return theorem

        return self._run(op)

    def upsert(self, data: dict) -> TheoremORM:
        """
        Insert or update by unique statement.
        """
        def op(s):
            existing = s.query(TheoremORM).filter_by(statement=data["statement"]).first()

            # Normalize embedding_id via scorable_embeddings
            if data.get("embedding_id") is not None:
                self._resolve_embedding_field(s, data)

            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
                obj = existing
                action = "TheoremUpdated"
            else:
                obj = TheoremORM(**data)
                s.add(obj)
                action = "TheoremInserted"

            if self.logger:
                self.logger.log(action, obj.to_dict())
            return obj

        return self._run(op)

    def bulk_add_theorems(self, items: List[dict]) -> List[TheoremORM]:
        def op(s):
            theorems = []
            for item in items:
                self._resolve_embedding_field(s, item)
                theorems.append(TheoremORM(**item))
            s.add_all(theorems)
            # flush once for all for perf; Base._run will manage commit/rollback
            return theorems

        return self._run(op)

    # ---------- Queries ----------

    def get_by_id(self, theorem_id: int) -> Optional[TheoremORM]:
        def op(s):
            return s.query(TheoremORM).filter_by(id=theorem_id).first()
        return self._run(op)

    def get_by_statement(self, statement: str) -> Optional[TheoremORM]:
        def op(s):
            return s.query(TheoremORM).filter_by(statement=statement).first()
        return self._run(op)

    def get_all(self, limit: int = 100) -> List[TheoremORM]:
        def op(s):
            q = s.query(TheoremORM)
            if self.default_order_by is not None:
                q = q.order_by(self.default_order_by)
            return q.limit(limit).all()
        return self._run(op)

    def delete_by_id(self, theorem_id: int) -> bool:
        def op(s):
            obj = s.query(TheoremORM).filter_by(id=theorem_id).first()
            if obj:
                s.delete(obj)
                if self.logger:
                    self.logger.log("TheoremDeleted", {"id": theorem_id})
                return True
            return False
        return self._run(op)

    def get_by_run_id(self, run_id: int) -> List[TheoremORM]:
        def op(s):
            return s.query(TheoremORM).filter_by(pipeline_run_id=run_id).all()
        return self._run(op)
