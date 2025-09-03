# stephanie/memory/base_embedding_store.py
import hashlib

import torch

from stephanie.memory import BaseStore
from stephanie.utils.lru_cache import SimpleLRUCache


class BaseEmbeddingStore(BaseStore):
    def __init__(self, cfg, conn, db, table: str, name: str, embed_fn, logger=None, cache_size=10000):
        super().__init__(db, logger)
        self.cfg = cfg
        self.conn = conn
        self.dim = cfg.get("dim", 1024)
        self.hdim = self.dim // 2
        self.table = table
        self.name = name
        self.type = cfg.get("type", name)  # e.g. "hnet", "hf"
        self.embed_fn = embed_fn

        # Cache: {hash -> (embedding_id, embedding_vector)}
        self._cache = SimpleLRUCache(max_size=cache_size)

    def name(self) -> str:
        return self.name

    def __repr__(self):
        return f"<{self.__class__.__name__} table={self.table} type={self.type}>"

    def get_text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_or_create(self, text: str):
        """Return embedding vector for text, caching both id + embedding."""
        text_hash = self.get_text_hash(text)

        cached = self._cache.get(text_hash)
        if cached:
            return cached[1]  # embedding vector

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, embedding FROM {self.table} WHERE text_hash = %s",
                    (text_hash,),
                )
                row = cur.fetchone()
                if row:
                    embedding_id, embedding = row
                    self._cache.set(text_hash, (embedding_id, embedding))
                    return embedding
        except Exception as e:
            if self.logger:
                self.logger.log("EmbeddingFetchFailed", {"error": str(e)})

        # Not found → create
        embedding = self.embed_fn(text, self.cfg)
        embedding_id = None
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.table} (text, text_hash, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (text_hash) DO NOTHING
                    RETURNING id;
                    """,
                    (text, text_hash, embedding),
                )
                row = cur.fetchone()
                if row:
                    embedding_id = row[0]
            self.conn.commit()
        except Exception as e:
            if self.logger:
                self.logger.log("EmbeddingInsertFailed", {"error": str(e)})

        # Fall back: lookup id if INSERT didn’t return
        if embedding_id is None:
            embedding_id = self.get_id_for_text(text)

        self._cache.set(text_hash, (embedding_id, embedding))
        return embedding

    def get_id_for_text(self, text: str) -> int | None:
        """Return embedding id for a text, cached if available."""
        text_hash = self.get_text_hash(text)
        cached = self._cache.get(text_hash)
        if cached:
            return cached[0]  # embedding_id

        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT id FROM {self.table} WHERE text_hash = %s", (text_hash,))
                row = cur.fetchone()
                if row:
                    embedding_id = row[0]
                    self._cache.set(text_hash, (embedding_id, None))  # no vector
                    return embedding_id
        except Exception as e:
            if self.logger:
                self.logger.log("EmbeddingIdFetchFailed", {"error": str(e)})
        return None

    def search_related_scorables(self, query: str, target_type: str = "document", top_k: int = 10, with_metadata: bool = True):
        """
        Search for scorables (documents, plan_traces, etc.) similar to the query.
        """
        try:
            query_emb = self.get_or_create(query)

            base_sql = """
                SELECT se.scorable_id, se.scorable_type, se.embedding_id,
                       1 - (e.embedding <-> %s::vector) AS score
            """
            join_sql = f"FROM scorable_embeddings se JOIN {self.table} e ON se.embedding_id = e.id"

            if with_metadata and target_type == "document":
                base_sql += ", d.title, d.summary, d.text"
                join_sql += " JOIN documents d ON se.scorable_id::int = d.id"

            sql = f"""
                {base_sql}
                {join_sql}
                WHERE se.scorable_type = %s
                ORDER BY e.embedding <-> %s::vector
                LIMIT %s;
            """

            with self.conn.cursor() as cur:
                cur.execute(sql, (query_emb, target_type, query_emb, top_k))
                rows = cur.fetchall()

            results = []
            for row in rows:
                base = {
                    "id": row[0],
                    "scorable_type": row[1],
                    "embedding_id": row[2],
                    "score": float(row[3]),
                }
                if with_metadata and target_type == "document":
                    base.update({"title": row[4], "summary": row[5], "text": row[6]})
                results.append(base)
            return results
        except Exception as e:
            if self.logger:
                self.logger.log("ScorableSearchFailed", {"error": str(e), "query": query})
            return []

    def find_neighbors(self, embedding, k: int = 5):
        """
        Given an embedding vector, return nearest neighbor texts from the table.
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().tolist()

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT e.id, e.text, 1 - (e.embedding <-> %s::vector) AS score
                    FROM {self.table} e
                    WHERE e.embedding IS NOT NULL
                    ORDER BY e.embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (embedding, embedding, k),
                )
                rows = cur.fetchall()
            return [
                {"id": row[0], "text": row[1], "score": float(row[2])}
                for row in rows
            ]
        except Exception as e:
            if self.logger:
                self.logger.log("FindNeighborsFailed", {"error": str(e)})
            return []
