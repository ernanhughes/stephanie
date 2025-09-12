# stephanie/utils/json_sanitize.py
import math
from typing import Any
import json
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