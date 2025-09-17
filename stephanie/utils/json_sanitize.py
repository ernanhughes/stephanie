# stephanie/utils/json_sanitize.py
from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any, Iterable, Mapping

import numpy as np


def _is_float(x: Any) -> bool:
    if isinstance(x, float):
        return True
    if isinstance(x, (np.floating,)):
        return True
    return False

def _is_int(x: Any) -> bool:
    if isinstance(x, int) and not isinstance(x, bool):
        return True
    if isinstance(x, (np.integer,)):
        return True
    return False

def json_sanitize(obj: Any) -> Any:
    """
    Recursively convert obj into JSON-safe primitives:
    - float NaN/Inf → None
    - numpy scalars → Python scalars
    - sets/tuples/ndarrays → lists
    Leaves dict keys/strings/bools/None as-is.
    """
    # scalars
    if obj is None or isinstance(obj, (str, bool)):
        return obj

    if _is_int(obj):
        return int(obj)

    if _is_float(obj):
        x = float(obj)
        return x if math.isfinite(x) else None

    # sequences
    if isinstance(obj, (list, tuple, set)):
        return [json_sanitize(v) for v in obj]

    import numpy as np
    if isinstance(obj, np.ndarray):
        return [json_sanitize(v) for v in obj.tolist()]

    # mappings
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}

    # fallback: best-effort string
    try:
        return str(obj)
    except Exception:
        return None


def safe_json(obj: Any) -> str:
    def default_serializer(x):
        try:
            return str(x)
        except Exception:
            return "<unserializable>"
    return json.dumps(obj, default=default_serializer, ensure_ascii=False)


def to_json_safe(obj: Any) -> Any:
    """Convert tuples/np types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def _to_native_scalar(x: Any) -> Any:
    # numpy → python
    if np is not None:
        if isinstance(x, (np.floating,)):
            val = float(x)
            return None if (math.isnan(val) or math.isinf(val)) else val
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if x is np.nan or x is np.inf or x is -np.inf:
            return None

    # plain python numbers
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, (int, bool, str)) or x is None:
        return x

    # datetimes
    if isinstance(x, (datetime, date)):
        return x.isoformat()

    # bytes
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return str(x)

    return x

def sanitize_for_json(obj: Any, *, max_depth: int = 100) -> Any:
    """
    Recursively convert obj into JSON-serializable python natives.
    - numpy scalars → python scalars
    - NaN/±Inf → None
    - sets/tuples → lists
    - datetimes → ISO strings
    - unknown objects → str(obj)
    """
    if max_depth <= 0:
        return str(obj)

    # Fast path for common scalars
    v = _to_native_scalar(obj)
    if v is not obj:
        return v
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj

    # Mappings/dicts
    if isinstance(obj, Mapping):
        out = {}
        for k, vv in obj.items():
            # force key to string
            kk = str(k)
            out[kk] = sanitize_for_json(vv, max_depth=max_depth - 1)
        return out

    # Iterables (lists, tuples, sets, generators)
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
        return [sanitize_for_json(x, max_depth=max_depth - 1) for x in obj]

    # Fallback
    try:
        json.dumps(obj)  # will raise if not serializable
        return obj
    except Exception:
        return str(obj)
