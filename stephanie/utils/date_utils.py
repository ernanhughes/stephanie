# stephanie/utils/date_utils.py
from __future__ import annotations

from datetime import datetime, timezone


def utcnow():
    return datetime.now(timezone.utc)

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def iso_date(dt: datetime | None) -> str | None:
    return dt.isoformat() if isinstance(dt, datetime) else None

