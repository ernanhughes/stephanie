from sqlalchemy.orm import Session
from datetime import datetime
from stephanie.models.entity_cache import EntityCacheORM
from stephanie.models.scorable_embedding import ScorableEmbeddingORM
import json


class EntityCacheStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "entity_cache"

    def upsert(self, embedding_ref, results: list[dict], *, embedding_type: str = "ner", scorable_id: str = None, scorable_type: str = "query") -> EntityCacheORM:
        """
        Upsert cache results for a given embedding_ref.
        
        Args:
            embedding_ref: Either an integer (id in scorable_embeddings)
                           OR a raw text/string hash (will auto-resolve/create embedding).
            results: List of search result dicts (will be stored as JSON).
            embedding_type: Backend type (ner, hf, hnet, ollama, etc.)
            scorable_id: Optional scorable identifier if embedding_ref is text
            scorable_type: Type of scorable (default: 'query')
        """
        try:
            # If we were given a string/text → resolve into ScorableEmbeddingORM row
            if isinstance(embedding_ref, str):
                # Hash it consistently
                from hashlib import sha256
                text_hash = sha256(embedding_ref.encode("utf-8")).hexdigest()

                # Try lookup first
                emb_row = (
                    self.session.query(ScorableEmbeddingORM)
                    .filter_by(scorable_id=scorable_id or text_hash,
                               scorable_type=scorable_type,
                               embedding_type=embedding_type)
                    .first()
                )

                if not emb_row:
                    # Create a placeholder row (no vector stored here)
                    emb_row = ScorableEmbeddingORM(
                        scorable_id=scorable_id or text_hash,
                        scorable_type=scorable_type,
                        embedding_id=-1,  # -1 = not linked to actual embedding backend yet
                        embedding_type=embedding_type,
                    )
                    self.session.add(emb_row)
                    self.session.commit()

                embedding_ref = emb_row.id

            # Now embedding_ref is definitely an integer
            cache_row = (
                self.session.query(EntityCacheORM)
                .filter_by(embedding_ref=embedding_ref)
                .first()
            )

            if cache_row:
                cache_row.results_json = results
                cache_row.last_updated = datetime.now()
            else:
                cache_row = EntityCacheORM(
                    embedding_ref=embedding_ref,
                    results_json=results,
                    last_updated=datetime.now()
                )
                self.session.add(cache_row)

            self.session.commit()
            return cache_row

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("EntityCacheUpsertError", {"error": str(e)})
            raise

    def get_by_embedding(self, embedding_ref: int | str) -> EntityCacheORM | None:
        """
        Fetch cache row by embedding reference (int id or text).
        """
        if isinstance(embedding_ref, str):
            # hash string → resolve through ScorableEmbeddingORM
            from hashlib import sha256
            text_hash = sha256(embedding_ref.encode("utf-8")).hexdigest()
            emb_row = (
                self.session.query(ScorableEmbeddingORM)
                .filter_by(scorable_id=text_hash)
                .first()
            )
            if not emb_row:
                return None
            embedding_ref = emb_row.id

        return (
            self.session.query(EntityCacheORM)
            .filter_by(embedding_ref=embedding_ref)
            .first()
        )
