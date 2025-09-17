# stephanie/utils/id_schema.py
from __future__ import annotations

import hashlib


def canonical_node_id(scorable_type: str, scorable_pk: str|int, text: str, start: int, end: int) -> str:
    """
    Stable, interpretable ID:
      <scorable_type>:<scorable_pk>:<12-char-hash>
    Full context (text/span) is kept in metadata, not the ID.
    """
    raw = f"{scorable_type}:{scorable_pk}:{text}:{start}:{end}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{scorable_type}:{scorable_pk}:{digest}"
