# stephanie/memory/generic_embedding_store.
from __future__ import annotations

import hashlib

import torch

from stephanie.memory import BaseStore
from stephanie.tools import get_embedding as get_ollama_embedding
# Backend map – each embedding type points to its generator
from stephanie.tools.hf_embedding import get_embedding as get_hf_embedding
from stephanie.tools.hnet_embedder import get_embedding as get_hnet_embedding
from stephanie.utils.lru_cache import SimpleLRUCache

BACKENDS = {
    "huggingface": get_hf_embedding,
    "hnet": get_hnet_embedding,
    "ollama": get_ollama_embedding,
}

class GenericEmbeddingStore(BaseStore):
    def __init__(self, cfg, conn, db, logger=None, cache_size=10000):
        super().__init__(db, logger)
        self.cfg = cfg
        self.conn = conn

        # Configure from cfg
        self.table = cfg.get("table", "embeddings")  # e.g. "hf_embeddings", "hnet_embeddings"
        self.type = cfg.get("type", "ollama")        # e.g. "huggingface", "hnet", "ollama"
        self.name = f"{self.type}_embeddings"
        self.dim = cfg.get("dim", 1024)
        self.hdim = self.dim // 2

        self._cache = SimpleLRUCache(max_size=cache_size)

        if self.type not in BACKENDS:
            raise ValueError(f"Unsupported embedding type: {self.type}")
        self._backend = BACKENDS[self.type]

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None} table={self.table} type={self.type}>"

    def _ensure_list(self, embedding):
        if isinstance(embedding, torch.Tensor):
            return embedding.detach().cpu().tolist()
        return embedding

    def get_text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_or_create(self, text: str):
        text_hash = self.get_text_hash(text)
        cached = self._cache.get(text_hash)
        if cached:
            return cached

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT embedding FROM {self.table} WHERE text_hash = %s",
                    (text_hash,),
                )
                row = cur.fetchone()
                if row:
                    embedding = self._ensure_list(row[0])
                    self._cache.set(text_hash, embedding)
                    return embedding
        except Exception as e:
            if self.logger:
                self.logger.log("EmbeddingFetchFailed", {"error": str(e)})

        embedding = self._ensure_list(self._backend(text, self.cfg))
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
        except Exception as e:
            if self.logger:
                self.logger.log("EmbeddingInsertFailed", {"error": str(e)})

        self._cache.set(text_hash, embedding)
        return embedding

    def get_id_for_text(self, text: str):
        text_hash = self.get_text_hash(text)
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT id FROM {self.table} WHERE text_hash = %s", (text_hash,)
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            if self.logger:
                self.logger.log("EmbeddingIdFetchFailed", {"error": str(e)})
            return None

    def find_neighbors(self, embedding, k: int = 5):
        embedding = self._ensure_list(embedding)
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, text, 1 - (embedding <-> %s::vector) AS score
                    FROM {self.table}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (embedding, embedding, k),
                )
                return [row[1] for row in cur.fetchall()]
        except Exception as e:
            if self.logger:
                self.logger.log("FindNeighborsFailed", {"error": str(e)})
            return []

    def search_related_scorables(
        self,
        query: str,
        top_k: int = 10,
        document_type: str = "document",
    ):
        """
        Search for documents of a given type using embeddings stored in document_embeddings.

        Args:
            query (str): The search query text.
            top_k (int): Number of results to return.
            document_type (str): Type of document ("document", "prompt", "hypothesis", etc.)

        Returns:
            list[dict]: Matching documents with scores.
        """
        try:
            # Create embedding for the query
            embedding = self.get_or_create(query)

            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT 
                        d.id,
                        d.title,
                        d.summary,
                        d.content,
                        de.embedding_id,
                        1 - (e.embedding <-> %s::vector) AS score
                    FROM document_embeddings de
                    JOIN documents d ON de.document_id::int = d.id
                    JOIN {self.table} e ON de.embedding_id = e.id
                    WHERE de.document_type = %s
                    AND de.embedding_type = %s
                    ORDER BY e.embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (embedding, document_type, self.name, embedding, top_k),
                )
                rows = cur.fetchall()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "summary": row[2],
                    "content": row[3],
                    "embedding_id": row[4],
                    "score": float(row[5]),
                    "text": row[2] or row[3],  # Prefer summary, fallback to content
                    "source": document_type,
                    "embedding_type": self.type,
                }
                for row in rows
            ]

        except Exception as e:
            if self.logger:
                self.logger.log(
                    "DocumentSearchFailed",
                    {"error": str(e), "query": query, "embedding_type": self.type},
                )
            else:
                print(f"[DocumentSearchFailed] {e}")
            return []
