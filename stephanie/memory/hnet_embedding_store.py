# stephanie/memory/hnet_embedding_store.py
import hashlib

from stephanie.memory import BaseStore
from stephanie.tools.hnet_embedder import get_embedding
from stephanie.utils.lru_cache import SimpleLRUCache


class HNetEmbeddingStore(BaseStore):
    def __init__(self, cfg, conn, db, logger=None, cache_size=10000):
        super().__init__(db, logger)
        self.cfg = cfg
        self.conn = conn
        self.dim = cfg.get("dim", 1024)  # Default to 1024 if not specified
        self.hdim = self.dim // 2
        self.name = "hnet_embeddings"
        self.table = "hnet_embeddings"
        self.type = "hnet"

        self._cache = SimpleLRUCache(max_size=cache_size)

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None} cfg={self.cfg}>"

    def name(self) -> str:
        return "hnet_embeddings"

    def get_or_create(self, text: str):
        text_hash = self.get_text_hash(text)

        cached = self._cache.get(text_hash)
        if cached:
            return cached

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT embedding FROM hnet_embeddings WHERE text_hash  = %s",
                    (text_hash,),
                )
                row = cur.fetchone()
                if row:
                    return row[0]  # Force conversion to list of floats
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("EmbeddingFetchFailed", {"error": str(e)})

        embedding = get_embedding(text, self.cfg)
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hnet_embeddings (text, text_hash, embedding)
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


    def search_similar_documents_with_scores(
        self, document: str, top_k: int = 5
    ) -> list:
        """
        Embedding-based search for similar prompts and their scores.
        Returns:
            A list of dicts: [{prompt, score, pipeline_run_id, step_index, dimension_scores}, ...]
        """
        embedding = get_embedding(document, self.cfg)

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        p.prompt_text AS prompt,
                        p.pipeline_run_id,
                        s.score,
                        s.dimension,
                        s.source 
                    FROM prompts p
                    JOIN hnet_embeddings x
                        ON x.id = p.embedding_id
                    JOIN evaluations e
                        ON e.pipeline_run_id = p.pipeline_run_id
                    JOIN scores s 
                        ON s.evaluation_id = e.id
                    WHERE x.embedding IS NOT NULL
                    ORDER BY x.embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (embedding, top_k),
                )
                rows = cur.fetchall()

            results = []
            for row in rows:
                results.append(
                    {
                        "prompt": row[0],
                        "pipeline_run_id": row[1],
                        "score": float(row[2]) if row[2] is not None else 0.0,
                        "dimension": row[3],
                        "source": row[4],
                    }
                )

            if self.logger:
                self.logger.log(
                    "PromptSimilaritySearch",
                    {
                        "query": document,
                        "top_k": top_k,
                        "returned": len(results),
                    },
                )

            return results

        except Exception as e:
            if self.logger:
                self.logger.log(
                    "PromptSearchFailed", {"query": document, "error": str(e)}
                )
            else:
                print(f"[PromptSearchFailed] {e}")
            return []

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
                        "SELECT id FROM hnet_embeddings WHERE text_hash = %s", (text_hash,)
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
                        "SELECT id FROM hnet_embeddings WHERE text_hash = %s", (text_hash,)
                    )
                    row = cur.fetchone()
                    return row[0] if row else None
            except Exception as e:
                if self.logger:
                    self.logger.log("EmbeddingIdFetchFailed", {"error": str(e)})
                return None

    def search_related_scorables(
        self,
        query: str,
        target_type: str = "document",
        top_k: int = 10,
        with_metadata: bool = True,
    ):
        """
        Generic search across any scorable type using pgvector.
        Assumes embeddings live in scorable_embeddings with (scorable_id, scorable_type).

        Args:
            query (str): Search query text.
            target_type (str): Scorable type ("document", "prompt", "plan_trace", etc.).
            top_k (int): Number of results to return.
            with_metadata (bool): If True, join against the target table for extra fields.

        Returns:
            list[dict]: Matching scorables with scores (and optional metadata).
        """
        try:
            embedding = self.get_or_create(query)

            base_sql = """
                SELECT 
                    se.scorable_id,
                    se.scorable_type,
                    se.embedding_id,
                    1 - (e.embedding <-> %s::vector) AS score
            """

            join_sql = f"""
                FROM scorable_embeddings se
                JOIN {self.table} e ON se.embedding_id = e.id
            """

            # Optional: join metadata if table exists for this scorable type
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
                cur.execute(sql, (embedding, target_type, embedding, top_k))
                rows = cur.fetchall()

            results = []
            for row in rows:
                base_result = {
                    "id": row[0],
                    "scorable_type": row[1],
                    "embedding_id": row[2],
                    "score": float(row[3]),
                }
                if with_metadata and target_type == "document":
                    base_result.update(
                        {"title": row[4], "summary": row[5], "text": row[6]}
                    )
                results.append(base_result)

            return results

        except Exception as e:
            if self.logger:
                self.logger.log("ScorableSearchFailed", {"error": str(e), "query": query})
            else:
                print(f"[VectorMemory] Scorable search failed: {e}")
            return []

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
                    FROM hnet_embeddings e
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
