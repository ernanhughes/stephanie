# stephanie/memory/hf_embedding_store.py

import hashlib

from stephanie.memory import BaseStore
from stephanie.tools.hf_embedding import get_embedding
from stephanie.utils.lru_cache import SimpleLRUCache


class HuggingFaceEmbeddingStore(BaseStore):
    def __init__(self, cfg, conn, db, logger=None):
        super().__init__(db, logger)
        self.cfg = cfg
        self.conn = conn
        self.name = "hf_embeddings"
        self.type = "huggingface"
        self.dim = cfg.get("dim", 1024)  # Default to 1024 if not specified
        self.hdim = self.dim // 2
        self.cache_size = cfg.get("cache_size", 10000)  # Default cache size
        self._cache = SimpleLRUCache(max_size=self.cache_size)

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None} cfg={self.cfg}>"

    def name(self) -> str:
        return "hf_embeddings"

    def get_or_create(self, text: str):
        text_hash = self.get_text_hash(text)

        cached = self._cache.get(text_hash)
        if cached:
            return cached

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT embedding FROM hf_embeddings WHERE text_hash = %s",
                    (text_hash,),
                )
                row = cur.fetchone()
                if row:
                    return row[0]
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("EmbeddingFetchFailed", {"error": str(e)})

        embedding = get_embedding(text, self.cfg)
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hf_embeddings (text, text_hash, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (text_hash) DO NOTHING
                    RETURNING text_hash;
                """,
                    (text, text_hash, embedding),
                )
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("EmbeddingInsertFailed", {"error": str(e)})
        self._cache.set(text_hash, embedding)
        return embedding

    def get_text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


    def get_id_for_text(self, text: str) -> int | None:
        """
        Retrieves the embedding ID for the given text if it exists in the database.
        Returns:
            The embedding ID (int) if found, otherwise None.
        """
        text_hash = self.get_text_hash(text)

        # First check the cache to avoid unnecessary DB hit
        cached = self._cache.get(text_hash)
        if cached:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM hf_embeddings WHERE text_hash = %s", (text_hash,)
                    )
                    row = cur.fetchone()
                    return row[0] if row else None
            except Exception as e:
                if self.logger:
                    self.logger.log("EmbeddingIdFetchFailed", {"error": str(e)})
                return None
        else:
            # No cache, still check DB
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM hf_embeddings WHERE text_hash = %s", (text_hash,)
                    )
                    row = cur.fetchone()
                    return row[0] if row else None
            except Exception as e:
                if self.logger:
                    self.logger.log("EmbeddingIdFetchFailed", {"error": str(e)})
                return None


    def find_neighbors(self, embedding: list[float], k: int = 5) -> list[str]:
        """
        Return the text associated with the k nearest neighbors to the given embedding.

        Args:
            embedding (list[float]): The embedding vector to compare against.
            k (int): Number of nearest neighbors to return.

        Returns:
            list[str]: List of text contents of the top-k nearest items.
        """
        try:
            import torch    
              # ✅ Fix: convert tensor to list if needed
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().tolist()
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        e.id,
                        e.text,
                        1 - (e.embedding <-> %s::vector) AS score  -- cosine similarity proxy
                    FROM hf_embeddings e
                    WHERE e.embedding IS NOT NULL
                    ORDER BY e.embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (embedding, embedding, k),
                )
                rows = cur.fetchall()

            return [row[1] for row in rows]  # Return only the 'text' column

        except Exception as e:
            if self.logger:
                self.logger.log("FindNeighborsFailed", {"error": str(e)})
            else:
                print(f"[EmbeddingStore] find_neighbors failed: {e}")
            return []


    def search_related_documents(self, query: str, top_k: int = 10):
        try:
            embedding = self.get_or_create(query)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        d.id,
                        d.title,
                        d.summary,
                        d.content,
                        d.embedding_id,
                        1 - (e.embedding <-> %s::vector) AS score  -- cosine similarity proxy
                    FROM documents d
                    JOIN hf_embeddings e ON d.embedding_id = e.id
                    WHERE d.embedding_id IS NOT NULL
                    ORDER BY e.embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (embedding, embedding, top_k),
                )
                results = cur.fetchall()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "summary": row[2],
                    "content": row[3],
                    "embedding_id": row[4],
                    "score": float(row[5]),
                    "text": row[2] or row[3],  # Default to summary, fallback to content
                    "source": "document",
                }
                for row in results
            ]

        except Exception as e:
            if self.logger:
                self.logger.log(
                    "DocumentSearchFailed", {"error": str(e), "query": query}
                )
            else:
                print(f"[VectorMemory] Document search failed: {e}")
            return []
