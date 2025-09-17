# stephanie/memory/entity_cache_store.py
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.entity_cache import EntityCacheORM
from stephanie.utils.json_sanitize import to_json_safe


class EntityCacheStore(BaseSQLAlchemyStore):
    """
    DB-backed cache for entity/neighbor search results.
    Keyed by scorable_embeddings.id (embedding_ref).
    """
    orm_model = EntityCacheORM
    default_order_by = EntityCacheORM.last_updated.desc()

    def __init__(self, session: Session, logger=None):
        super().__init__(session, logger)
        self.name = "entity_cache"

    def name(self) -> str:
        return self.name

    def get_by_embedding(self, embedding_ref: int) -> Optional[EntityCacheORM]:
        try:
            return (
                self.session.query(EntityCacheORM)
                .filter(EntityCacheORM.embedding_ref == embedding_ref)
                .one_or_none()
            )
        except SQLAlchemyError as e:
            if self.logger:
                self.logger.log("EntityCacheFetchFailed", {"error": str(e), "embedding_ref": embedding_ref})
            return None

    def upsert(self, embedding_ref: int, results_json: Any) -> EntityCacheORM:
        """
        Insert or update cache row for this embedding_ref.
        results_json must be JSON-serializable (convert np types to Python).
        """
        try:
            row = (
                self.session.query(EntityCacheORM)
                .filter(EntityCacheORM.embedding_ref == embedding_ref)
                .one_or_none()
            )
            if row is None:
                row = EntityCacheORM(
                    embedding_ref=embedding_ref,
                    results_json=to_json_safe(results_json),
                    last_updated=datetime.now(),
                )
                self.session.add(row)
            else:
                row.results_json = to_json_safe(results_json)
                row.last_updated = datetime.now()

            self.session.commit()
            return row
        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("EntityCacheUpsertFailed", {"error": str(e), "embedding_ref": embedding_ref})
            raise

