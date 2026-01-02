# stephanie/components/information/paper/pipeline/ids.py
from __future__ import annotations

from typing import Any, Optional


def make_section_scorable_id(*, arxiv_id: str, section: Any, fallback_index: Optional[int] = None) -> str:
    """
    Canonical, stable section scorable id.

    Priority:
      1) If the section already has a stable id, use it:
         - section.scorable_id
         - section.section_id
         - section.id
      2) Else fall back to an index-based stable id:
         "{arxiv_id}::section::{index}"

    Notes:
      - This mirrors what your old agent did implicitly in several places.
      - Keeping it centralized prevents subtle mismatches across pipeline stages.
    """
    arxiv_id = str(arxiv_id)

    sid = getattr(section, "scorable_id", None) or getattr(section, "section_id", None) or getattr(section, "id", None)
    if sid:
        return str(sid)

    idx = getattr(section, "index", None)
    if idx is None:
        idx = fallback_index if fallback_index is not None else 0

    return f"{arxiv_id}::section::{int(idx)}"
