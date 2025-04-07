import psycopg2
from psycopg2.extras import RealDictCursor
from hydra import initialize, compose
from models import MemoryEntry

import logging
logger = logging.getLogger(__name__)


def connect_db():
    logger.info("Connecting to db...")
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="stephanie.yaml")
        db = cfg.db
        return psycopg2.connect(
            dbname=db.name,
            user=db.user,
            password=db.password,
            host=db.host,
            port=db.port
        )

def fetch_memory_by_id(memory_id: int) -> MemoryEntry | None:
    conn = connect_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM memory WHERE id = %s", (memory_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return MemoryEntry(**row)
    return None

def update_memory_fields(memory_id: int, data: dict) -> bool:
    allowed_fields = {"summary", "tags"}
    updates = {k: data[k] for k in allowed_fields if k in data}
    if not updates:
        return False

    set_clause = ", ".join(f"{k} = %s" for k in updates)
    values = list(updates.values()) + [memory_id]

    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        f"UPDATE memory SET {set_clause} WHERE id = %s", values
    )
    success = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return success
