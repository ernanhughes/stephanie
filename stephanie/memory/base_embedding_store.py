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
        self.name  = name
        self.type = cfg.get("type", name)  # e.g. "hnet", "hf"
        self.embed_fn = embed_fn

        # Cache: {hash -> (id, embedding)}
        self._cache = SimpleLRUCache(max_size=cache_size)

    def name(self) -> str:
        return self.name

    def __repr__(self):
        return f"<{self.__class__.__name__} table={self.table} type={self.type}>"

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

        embedding = self.get_embed_fn(text, self.cfg)
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

    def get_id_for_text(self, text: str) -> int | None:
        text_hash = self.get_text_hash(text)
        cached = self._cache.get(text_hash)
        if cached:
            return cached[0]

        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT id FROM {self.table} WHERE text_hash = %s", (text_hash,))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            if self.logger:
                self.logger.log("EmbeddingIdFetchFailed", {"error": str(e)})
            return None

    def search_related_scorables(self, query: str, target_type: str = "document", top_k: int = 10, with_metadata: bool = True):
        try:
            _, query_emb = self.get_or_create(query)

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
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().tolist()

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT e.text, 1 - (e.embedding <-> %s::vector) AS score
                    FROM {self.table} e
                    WHERE e.embedding IS NOT NULL
                    ORDER BY e.embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (embedding, embedding, k),
                )
                rows = cur.fetchall()
            return [row[0] for row in rows]
        except Exception as e:
            if self.logger:
                self.logger.log("FindNeighborsFailed", {"error": str(e)})
            return []
