# stephanie/memory/entity_cache_store.py
from datetime import datetime
from typing import Any, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.entity_cache import EntityCacheORM
from stephanie.utils.json_sanitize import to_json_safe


class EntityCacheStore(BaseSQLAlchemyStore):
    """
    DB-backed cache for entity/neighbor search results.
    Keyed by scorable_embeddings.id (embedding_ref).
    """
    orm_model = EntityCacheORM
    default_order_by = "last_updated"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "entity_cache"

    def get_by_embedding(self, embedding_ref: int) -> Optional[EntityCacheORM]:
        def op(s):
            return s.query(EntityCacheORM).filter(EntityCacheORM.embedding_ref == embedding_ref).one_or_none()
        return self._run(op)

    def upsert(self, embedding_ref: int, results_json: Any) -> EntityCacheORM:
        """
        Insert or update cache row for this embedding_ref.
        results_json must be JSON-serializable (convert np types to Python).
        """
        def op(s):
            
                row = (
                    s.query(EntityCacheORM)
                    .filter(EntityCacheORM.embedding_ref == embedding_ref)
                    .one_or_none()
                )
                if row is None:
                    row = EntityCacheORM(
                        embedding_ref=embedding_ref,
                        results_json=to_json_safe(results_json),
                        last_updated=datetime.now(),
                    )
                    s.add(row)
                else:
                    row.results_json = to_json_safe(results_json)
                    row.last_updated = datetime.now()
                s.flush()
                return row

        return self._run(op)
