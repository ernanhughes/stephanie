# stephanie/utils/time_utils.py
from __future__ import annotations

from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_friendly_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()