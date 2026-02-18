import hashlib
import sqlite3
import time
from pathlib import Path
from typing import List

import numpy as np


class EmbeddingStore:

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        self._init_pragmas()
        self._init_schema()

    # -------------------------------------------------
    # Schema
    # -------------------------------------------------
    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(text_hash, model)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hash_model
            ON embeddings(text_hash, model)
        """)
        self.conn.commit()

    # -------------------------------------------------
    # Pragmas
    # -------------------------------------------------
    def _init_pragmas(self):
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.execute("PRAGMA mmap_size=30000000000")
        cur.close()

    # -------------------------------------------------
    # Hashing
    # -------------------------------------------------
    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # -------------------------------------------------
    # Fetch
    # -------------------------------------------------
    def get(self, texts: List[str], model: str):
        if not texts:
            return [], []

        hashes = [self._hash_text(t) for t in texts]

        rows = {}
        chunk = 900

        for i in range(0, len(hashes), chunk):
            sub = hashes[i:i+chunk]
            q = ",".join(["?"] * len(sub))

            sql = f"""
                SELECT text_hash, vec, dim
                FROM embeddings
                WHERE model = ?
                  AND text_hash IN ({q})
            """

            cur = self.conn.cursor()
            cur.execute(sql, [model, *sub])
            for r in cur.fetchall():
                rows[r["text_hash"]] = r
            cur.close()

        vecs = []
        missing_idx = []

        for i, h in enumerate(hashes):
            row = rows.get(h)
            if row is None:
                vecs.append(None)
                missing_idx.append(i)
                continue

            dim = int(row["dim"])
            v = np.frombuffer(row["vec"], dtype=np.float32)

            if v.shape[0] != dim:
                vecs.append(None)
                missing_idx.append(i)
                continue

            vecs.append(v)

        return vecs, missing_idx

    # -------------------------------------------------
    # Insert
    # -------------------------------------------------
    def put(self, texts: List[str], vecs: np.ndarray, model: str):
        now = time.time()
        cur = self.conn.cursor()

        for text, vec in zip(texts, vecs):
            text_hash = self._hash_text(text)

            cur.execute("""
                INSERT OR REPLACE INTO embeddings
                (text, text_hash, model, dim, vec, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                text,
                text_hash,
                model,
                int(vec.shape[0]),
                vec.astype(np.float32).tobytes(),
                now,
            ))

        self.conn.commit()
        cur.close()
