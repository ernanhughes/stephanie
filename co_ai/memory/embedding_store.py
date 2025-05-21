from co_ai.memory import BaseStore
from co_ai.tools import get_embedding

import pgvector.psycopg2
import hashlib

class EmbeddingStore(BaseStore):
    def __init__(self, db, cfg, logger=None):
        self.db = db
        self.cfg = cfg
        self.logger = logger
        self.name = "embedding"
        pgvector.psycopg2.register_vector(self.db)

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None} cfg={self.cfg}>"

    def name(self) -> str:
        return "embedding"

    def get_or_create(self, text: str):
        text_hash = self.get_text_hash(text)
        try:
            with self.db.cursor() as cur:
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
            with self.db.cursor() as cur:
                cur.execute("INSERT INTO embeddings (text, text_hash, embedding) VALUES (%s, %s, %s) ON CONFLICT (text_hash) DO NOTHING",
                            (text, text_hash, embedding))
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("EmbeddingInsertFailed", {"error": str(e)})
        return embedding
    
    def search_related(self, query: str, top_k: int = 5):
        try:
            embedding = get_embedding(query, self.cfg)
            with self.db.cursor() as cur:
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
                self.logger.log("HypothesesSearchFailed", {
                    "error": str(e),
                    "query": query
                })
            else:
                print(f"[VectorMemory] Search failed: {e}")
            return []

    def get_text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()