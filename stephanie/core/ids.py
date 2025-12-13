# stephanie/core/ids.py
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

_TIMESTAMP_RE = re.compile(r"^(\d{17})-([A-Z0-9_]+)-([0-9a-f]{8})$")


@dataclass(frozen=True, order=True)
class UniversalID:
    """
    Time-sortable universal ID for Stephanie.

    Format (string):
        YYYYMMDDHHMMSSmmm-KIND-randhex

    Example:
        20251206181530123-TASK-a3f9c1b2

    Properties:
        - Lexicographically sortable by time
        - Encodes UTC datetime down to milliseconds
        - 'kind' lets you namespace IDs (TASK, MEMCUBE, IDEA, etc.)
    """

    # Used for comparisons / sorting
    sort_key: int

    # Full string representation
    raw: str

    @classmethod
    def new(cls, kind: str = "GEN", when: Optional[datetime] = None) -> "UniversalID":
        """
        Create a new UniversalID.

        Args:
            kind: Short uppercase tag describing the entity ("TASK", "MEMCUBE", ...)
            when: Optional datetime. If omitted, uses now() in UTC.

        Returns:
            UniversalID instance
        """
        if when is None:
            when = datetime.now(timezone.utc)
        else:
            # Normalize to UTC
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            else:
                when = when.astimezone(timezone.utc)

        # 17-digit timestamp: YYYYMMDDHHMMSSmmm (milliseconds)
        ts_str = when.strftime("%Y%m%d%H%M%S%f")[:-3]  # strip microseconds -> ms
        sort_key = int(ts_str)

        # 32 bits of randomness (8 hex chars) – enough for most app-level IDs
        rand = secrets.token_hex(4)

        kind = kind.upper()
        raw = f"{ts_str}-{kind}-{rand}"
        return cls(sort_key=sort_key, raw=raw)

    @classmethod
    def parse(cls, raw: str) -> "UniversalID":
        """
        Reconstruct a UniversalID from its string form.
        """
        m = _TIMESTAMP_RE.match(raw)
        if not m:
            raise ValueError(f"Invalid UniversalID format: {raw!r}")
        ts_str, _kind, _rand = m.groups()
        sort_key = int(ts_str)
        return cls(sort_key=sort_key, raw=raw)

    @property
    def datetime(self) -> datetime:
        """
        Extract the UTC datetime encoded in this ID.
        """
        m = _TIMESTAMP_RE.match(self.raw)
        if not m:
            raise ValueError(f"Invalid UniversalID format: {self.raw!r}")

        ts_str = m.group(1)
        # YYYYMMDDHHMMSSmmm
        year = int(ts_str[0:4])
        month = int(ts_str[4:6])
        day = int(ts_str[6:8])
        hour = int(ts_str[8:10])
        minute = int(ts_str[10:12])
        second = int(ts_str[12:14])
        millisecond = int(ts_str[14:17])

        return datetime(
            year, month, day,
            hour, minute, second,
            millisecond * 1000,
            tzinfo=timezone.utc,
        )

    @property
    def kind(self) -> str:
        """
        Return the KIND part (TASK, MEMCUBE, etc.).
        """
        m = _TIMESTAMP_RE.match(self.raw)
        if not m:
            raise ValueError(f"Invalid UniversalID format: {self.raw!r}")
        return m.group(2)

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:
        return f"UniversalID(raw={self.raw!r})"
