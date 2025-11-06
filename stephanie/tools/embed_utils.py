
import math
from typing import Any, List
import numpy as np


def cos_safe(a: Any, b: Any) -> float:
    va = as_list_floats(a)
    vb = as_list_floats(b)
    if not va or not vb:
        return 0.0
    s = sum(x * y for x, y in zip(va, vb))
    la = math.sqrt(sum(x * x for x in va)) or 1.0
    lb = math.sqrt(sum(x * x for x in vb)) or 1.0
    return s / (la * lb)



def as_list_floats(x: Any) -> List[float]:
    """Convert various vector representations into a plain List[float]."""
    if x is None:
        return []
    # Unwrap datatypes commonly seen from DB/drivers/caches
    if isinstance(x, np.ndarray):
        try:
            return [float(v) for v in x.ravel().tolist()]
        except Exception:
            return []
    if isinstance(x, (list, tuple)):
        try:
            return [float(v) for v in x]
        except Exception:
            return []
    # Some drivers might return memoryview/bytes (pgvector variants). Try best-effort.
    if isinstance(x, memoryview):
        try:
            # Often memoryview -> bytes of floats is not trivial; bail out safely.
            return []
        except Exception:
            return []
    # Unknown type
    return []


def has_vec(x: Any) -> bool:
    return len(as_list_floats(x)) > 0

