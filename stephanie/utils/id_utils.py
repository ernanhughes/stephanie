# stephanie/utils/id_schema.py
from __future__ import annotations

import hashlib
from typing import Any


def make_numeric_id(run_id: Any, label: str, index: int) -> int:
    """
    Build a sortable integer ID:
        [run_code][label_code][index:04d]
    Ensures ascending order by run → label → index.
    """
    # 1️⃣ derive a 4-digit run code from the run_id (UUID-safe)
    if isinstance(run_id, str):
        run_hash = abs(hash(run_id)) % 10_000
    else:
        run_hash = int(run_id) % 10_000

    # 2️⃣ map label to a stable small integer
    label_code = {"good": 1, "medium": 2, "opposite": 3}.get(label, 9)

    # 3️⃣ combine into a single sortable integer
    numeric_id = int(f"{run_hash:04d}{label_code}{index:04d}")
    return numeric_id


def canonical_node_id(scorable_type: str, scorable_pk: str|int, text: str, start: int, end: int) -> str:
    """
    Stable, interpretable ID:
      <scorable_type>:<scorable_pk>:<12-char-hash>
    Full context (text/span) is kept in metadata, not the ID.
    """
    raw = f"{scorable_type}:{scorable_pk}:{text}:{start}:{end}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{scorable_type}:{scorable_pk}:{digest}"
