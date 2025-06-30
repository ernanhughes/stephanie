import hashlib

from co_ai.memory import BaseStore
from co_ai.tools import get_embedding
from co_ai.utils.lru_cache import SimpleLRUCache


class EmbeddingStore(BaseStore):
    def __init__(self, cfg, conn, db, logger=None, cache_size=10000):
        super().__init__(db, logger)
        self.cfg = cfg
        self.conn = conn
        self.name = "embedding"
        self._cache = SimpleLRUCache(max_size=cache_size)


    def __repr__(self):
        return f"<{self.name} connected={self.db is not None} cfg={self.cfg}>"

    def name(self) -> str:
        return "embedding"

    def get_or_create(self, text: str):
        text_hash = self.get_text_hash(text)

        cached = self._cache.get(text_hash)
        if cached:
            return cached

        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT embedding FROM embeddings WHERE text_hash  = %s", (text_hash,))
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
                    INSERT INTO embeddings (text, text_hash, embedding)
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


    def search_related(self, query: str, top_k: int = 5):
        try:
            embedding = get_embedding(query, self.cfg)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                        SELECT 
                            h.text,
                            g.goal_text AS goal,
                            h.confidence,
                            h.review
                        FROM hypotheses h
                        JOIN goals g ON h.goal_id = g.id
                        ORDER BY h.embedding <-> %s
                        LIMIT %s;
                    """,
                    (embedding, top_k)
                )
                results = cur.fetchall()

            if self.logger:
                self.logger.log("HypothesesSearched", {
                    "query": query,
                    "top_k": top_k,
                    "result_count": len(results)
                })

            return results
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "HypothesesSearchFailed", {"error": str(e), "query": query}
                )
            else:
                print(f"[VectorMemory] Search failed: {e}")
            return []

    def search_similar_prompts_with_scores(self, query: str, top_k: int = 5) -> list:
        """
        Embedding-based search for similar prompts and their scores.
        Returns:
            A list of dicts: [{prompt, score, pipeline_run_id, step_index, dimension_scores}, ...]
        """
        embedding = get_embedding(query, self.cfg)

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
                    JOIN embeddings x
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
                        "source": row[4]
                    }
                )

            if self.logger:
                self.logger.log(
                    "PromptSimilaritySearch",
                    {
                        "query": query,
                        "top_k": top_k,
                        "returned": len(results),
                    },
                )

            return results

        except Exception as e:
            if self.logger:
                self.logger.log("PromptSearchFailed", {"query": query, "error": str(e)})
            else:
                print(f"[PromptSearchFailed] {e}")
            return []

    def get_text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def backfill_prompt_embeddings(self):
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, prompt_text FROM prompts WHERE embedding_id IS NULL"
            )
            rows = cur.fetchall()

        for prompt_id, prompt_text in rows:
            embedding = self.get_or_create(prompt_text)
            text_hash = self.get_text_hash(prompt_text)
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM embeddings WHERE text_hash = %s", (text_hash,)
                )
                embedding_row = cur.fetchone()
                if embedding_row:
                    embedding_id = embedding_row[0]
                    with self.conn.cursor() as cur2:
                        cur2.execute(
                            "UPDATE prompts SET embedding_id = %s WHERE id = %s",
                            (embedding_id, prompt_id),
                        )


    def search_similar_documents_with_scores(self, document: str, top_k: int = 5) -> list:
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
                    JOIN embeddings x
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
                        "source": row[4]
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
                self.logger.log("PromptSearchFailed", {"query": document, "error": str(e)})
            else:
                print(f"[PromptSearchFailed] {e}")
            return []
