# stephanie/utils/coerce_utils.py
from __future__ import annotations

def to_float(x, default: float = 0.0) -> float:
    """Safely cast x to float, tolerating None/''/invalid; returns default on failure."""
    try:
        if x is None:
            return float(default)
        if isinstance(x, str) and not x.strip():
            return float(default)
        return float(x)
    except Exception:
        return float(default)
