# sis/arena_emitters.py
from __future__ import annotations
from typing import Dict, Any
import sqlite3
import json
import time
import contextlib

class SqliteArenaEmitter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS arena_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    event TEXT,
                    payload TEXT,
                    t REAL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def __call__(self, payload: Dict[str, Any]):
        # sync-friendly callable; KnowledgeArena accepts sync or async
        run_id = payload.get("run_id") or ""
        ev = payload.get("event") or "unknown"
        t = payload.get("t") or time.time()
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("INSERT INTO arena_events (run_id,event,payload,t) VALUES (?,?,?,?)",
                         (run_id, ev, json.dumps(payload, ensure_ascii=False), t))
            conn.commit()
