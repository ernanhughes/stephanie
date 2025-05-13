from co_ai.memory import BaseStore
from co_ai.tools import get_embedding


class EmbeddingStore(BaseStore):
    def __init__(self, db, cfg, logger=None):
        self.db = db
        self.cfg = cfg
        self.logger = logger
        self.name = "embedding"

    def name(self) -> str:
        return "embedding"

    def get_or_create(self, text):
        try:
            with self.db as cur:
                cur.execute("SELECT embedding FROM embeddings WHERE text = %s", (text,))
                row = cur.fetchone()
                if row:
                    return row[0]
        except Exception as e:
            if self.logger:
                self.logger.log("EmbeddingFetchFailed", {"error": str(e)})

        embedding = get_embedding(text, self.cfg)
        try:
            with self.db as cur:
                cur.execute("INSERT INTO embeddings (text, embedding) VALUES (%s, %s) ON CONFLICT (text) DO NOTHING",
                            (text, embedding))
        except Exception as e:
            if self.logger:
                self.logger.log("EmbeddingInsertFailed", {"error": str(e)})
        return embedding
    
    def search_related(self, query: str, top_k: int = 5):
        try:
            embedding = get_embedding(query, self.cfg)
            with self.db as cur:
                cur.execute(
                    """
                    SELECT text, goal, confidence, review
                    FROM hypotheses
                    ORDER BY embedding <-> %s
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
