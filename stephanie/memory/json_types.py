from __future__ import annotations

import json
from typing import Any

from sqlalchemy.types import TEXT, TypeDecorator


class JsonSafe(TypeDecorator):
    """Dialect-agnostic JSON storage.
    Stores as TEXT with JSON serialization. Works on SQLite/DuckDB/Postgres.
    Use when you want strict portability without JSON/JSONB.
    """
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value: Any, dialect):
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def process_result_value(self, value: Any, dialect):
        if value is None:
            return None
        return json.loads(value)

