import psycopg2
from hydra import initialize, compose
from omegaconf import OmegaConf
from embed_and_store import get_embedding

import logging
logger = logging.getLogger(__name__)

def connect_db():
    logger.info("connecting")
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="stephanie.yaml")
        print(OmegaConf.to_yaml(cfg))
        db = cfg.db
        return psycopg2.connect(
            dbname=db.name,
            user=db.user,
            password=db.password,
            host=db.host,
            port=db.port,
        )

def semantic_search(query, top_k=10):
    logger.info("semantic_search")
    embedding = get_embedding(query)
    if embedding is None:
        return []

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, timestamp, user_text, ai_text, summary, tags, source, openai_url,
               1 - (embedding <#> %s) AS score
        FROM memory
        ORDER BY embedding <#> %s
        LIMIT %s
    """, (embedding, embedding, top_k))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [row_to_dict(row) for row in rows]

def full_text_search(query, top_k=10):
    logger.info("full_text_search")
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, timestamp, user_text, ai_text, summary, tags, source, openai_url,
               ts_rank_cd(to_tsvector('english', title || ' ' || summary || ' ' || tags::text), plainto_tsquery(%s)) AS score
        FROM memory
        WHERE to_tsvector('english', title || ' ' || summary || ' ' || tags::text) @@ plainto_tsquery(%s)
        ORDER BY score DESC
        LIMIT %s
    """, (query, query, top_k))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [row_to_dict(row) for row in rows]

def hybrid_search(query, top_k=10):
    embedding = get_embedding(query)
    if embedding is None:
        return []

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, timestamp, user_text, ai_text, summary, tags, source, openai_url,
               (1 - (embedding <#> %s)) AS vector_score,
               ts_rank_cd(to_tsvector('english', title || ' ' || summary || ' ' || tags::text), plainto_tsquery(%s)) AS text_score
        FROM memory
        WHERE to_tsvector('english', title || ' ' || summary || ' ' || tags::text) @@ plainto_tsquery(%s)
        ORDER BY ((1 - (embedding <#> %s)) * %s + ts_rank_cd(to_tsvector('english', title || ' ' || summary || ' ' || tags::text), plainto_tsquery(%s)) * %s) DESC
        LIMIT %s
    """, (
        embedding, query, query,
        embedding, cfg.query.hybrid_weights.vector,
        query, cfg.query.hybrid_weights.text,
        top_k
    ))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [row_to_dict(row) for row in rows]

def row_to_dict(row):
    return {
        "id": row[0],
        "title": row[1],
        "timestamp": row[2],
        "user_text": row[3],
        "ai_text": row[4],
        "summary": row[5],
        "tags": row[6],
        "source": row[7],
        "openai_url": row[8],
        "score": row[9]
    }
