from app.db import connect_db
from app.models import MemoryEntry
from app.embedding import get_embedding
from typing import List
from hydra import initialize, compose
from psycopg2.extras import RealDictCursor

# Load config once at module level
with initialize(config_path="../config", version_base=None):
    cfg = compose(config_name="stephanie.yaml")
    top_k = cfg.query.top_k
    weights = cfg.query.hybrid_weights

def run_search(query: str, mode: str = "hybrid") -> List[MemoryEntry]:
    if mode == "vector":
        return vector_search(query)
    elif mode == "text":
        return text_search(query)
    else:
        return hybrid_search(query)

def vector_search(query: str) -> List[MemoryEntry]:
    embedding = get_embedding(query)
    if embedding is None:
        return []

    conn = connect_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT *, 1 - (embedding <#> %s::vector) AS score
        FROM memory
        ORDER BY embedding <#> %s::vector
        LIMIT %s
    """, (embedding, embedding, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [MemoryEntry(**r) for r in rows]

def text_search(query: str) -> List[MemoryEntry]:
    conn = connect_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT *,
        ts_rank_cd(to_tsvector('english', title || ' ' || summary || ' ' || tags::text), plainto_tsquery(%s)) AS score
        FROM memory
        WHERE to_tsvector('english', title || ' ' || summary || ' ' || tags::text) @@ plainto_tsquery(%s)
        ORDER BY score DESC
        LIMIT %s
    """, (query, query, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [MemoryEntry(**r) for r in rows]

def hybrid_search(query: str) -> List[MemoryEntry]:
    embedding = get_embedding(query)
    if embedding is None:
        return []

    conn = connect_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT *,
        1 - (embedding <#> %s::vector) AS vector_score,
        ts_rank_cd(to_tsvector('english', title || ' ' || summary || ' ' || tags::text), plainto_tsquery(%s)) AS text_score
        FROM memory
        WHERE to_tsvector('english', title || ' ' || summary || ' ' || tags::text) @@ plainto_tsquery(%s)
        ORDER BY (%s * (1 - (embedding <#> %s::vector)) + %s * ts_rank_cd(to_tsvector('english', title || ' ' || summary || ' ' || tags::text), plainto_tsquery(%s))) DESC
        LIMIT %s
    """, (
        embedding,
        query, query,
        weights.vector,
        embedding,
        weights.text,
        query,
        top_k
    ))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [MemoryEntry(**r) for r in rows]
