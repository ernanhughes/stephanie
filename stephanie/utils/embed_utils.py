# stephanie/utils/embed_utils.py
from __future__ import annotations

import json
import math
from typing import Any, List

from typing import Callable, Optional
import random

import numpy as np  # optional; code works if NumPy isn't installed


def as_list_floats(x: Any) -> List[float]:
    """
    Robustly coerce various vector representations into List[float].

    Accepts:
      - list/tuple of numbers
      - numpy.ndarray (any shape → 1D ravel)
      - bytes/bytearray/memoryview containing JSON like "[...]" (pgvector/jsonb cases)
      - str that looks like a JSON list "[...]" (last resort)

    Returns [] on any failure.
    """
    if x is None:
        return []

    # numpy array
    if np is not None and isinstance(x, np.ndarray):
        try:
            return [float(v) for v in x.ravel().tolist()]
        except Exception:
            return []

    # python sequences
    if isinstance(x, (list, tuple)):
        try:
            return [float(v) for v in x]
        except Exception:
            return []

    # raw buffers (pgvector/jsonb often lands here)
    if isinstance(x, (bytes, bytearray, memoryview)):
        try:
            buf = x.tobytes() if isinstance(x, memoryview) else bytes(x)
            s = buf.decode("utf-8", errors="ignore").strip()
            if s.startswith("[") and s.endswith("]"):
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [float(v) for v in arr]
        except Exception:
            return []

    # stringified JSON list
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [float(v) for v in arr]
            except Exception:
                return []

    # Unknown type
    return []


def has_vec(x: Any) -> bool:
    """True iff x coerces to a non-empty float vector."""
    return len(as_list_floats(x)) > 0


def l2_norm(v: Any) -> float:
    """L2 norm with safe [] handling (returns 1.0 sentinel for zero vectors)."""
    vv = as_list_floats(v)
    if not vv:
        return 1.0
    s = 0.0
    for t in vv:
        s += t * t
    return math.sqrt(s) or 1.0


def cos_safe(a: Any, b: Any) -> float:
    """
    Cosine similarity that:
      - never uses ndarray truthiness
      - tolerates length mismatch (zip truncates)
      - returns 0.0 if either side is empty
    """
    va = as_list_floats(a)
    vb = as_list_floats(b)
    if not va or not vb:
        return 0.0
    num = sum(x * y for x, y in zip(va, vb))
    return num / (l2_norm(va) * l2_norm(vb))


def normalize_unit(v: Any) -> List[float]:
    """Return unit-norm version of v; [] stays []."""
    vv = as_list_floats(v)
    if not vv:
        return []
    n = l2_norm(vv)
    return [t / n for t in vv]


def topk_by_cos(items, target_vec, vec_of, k: int) -> List[int]:
    """
    Rank items by cosine similarity to target_vec.
    vec_of(i) -> any vector-like for item i.
    Returns indices of top-k items.
    """
    tgt = as_list_floats(target_vec)
    scored = []
    for i in range(len(items)):
        scored.append((cos_safe(vec_of(i), tgt), i))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [i for _, i in scored[: max(0, k)]]


def dist(a: Any, b: Any, metric: str = "cosine") -> float:
    """
    Distance between two vectors with safe coercion.
      metric="cosine"   → 1 - cosine_similarity  ∈ [0, 2]
      metric="angular"  → arccos(cos_sim)/π      ∈ [0, 1]
      metric="euclidean"→ L2 distance

    Returns 1.0 when either side is empty (neutral-ish fallback).
    """
    va = as_list_floats(a)
    vb = as_list_floats(b)
    if not va or not vb:
        return 1.0

    m = metric.lower()
    if m in ("cos", "cosine"):
        c = cos_safe(va, vb)
        # clamp for numerical stability
        if c > 1.0:
            c = 1.0
        if c < -1.0:
            c = -1.0
        return 1.0 - c

    if m in ("angular", "angle"):
        c = cos_safe(va, vb)
        c = max(-1.0, min(1.0, c))
        return math.acos(c) / math.pi

    if m in ("l2", "euclidean"):
        # handle length mismatch by including leftover dims
        n = min(len(va), len(vb))
        s = sum((va[i] - vb[i]) ** 2 for i in range(n))
        if len(va) > n:
            s += sum(x * x for x in va[n:])
        elif len(vb) > n:
            s += sum(y * y for y in vb[n:])
        return math.sqrt(s)

    # default
    return 1.0 - cos_safe(va, vb)


def _cosine_dist_vec(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    # a: (n,d), b: (d,)
    # vectors assumed unit-normalized
    sim = a @ b  # (n,)
    # clamp to avoid tiny numeric spill
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def farthest_point_sample(
    indexes: List[int],
    vecs: List[List[float]],
    k: int,
    seed: int = 0,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[int]:
    """
    Greedy max-min (k-center) on `indexes` using cosine distance.
    - Vectorized with NumPy when available: O(k·n) distance work.
    - Fallback is pure-Python but still O(k·n) via a maintained min-distance array.
    - `progress(step, total)` is called each iteration (step ∈ [1..k]).
    """
    if not indexes:
        return []
    if k <= 0:
        return []
    if k == 1:
        rng = random.Random(seed)
        return [rng.choice(indexes)]

    rng = random.Random(seed)

    # ------- Fast path: NumPy vectorized -------
    if np is not None:
        # Build dense matrix only for the subset `indexes`
        sub = [normalize_unit(vecs[i]) for i in indexes]
        if not sub or not sub[0]:
            # fallback: degenerate input -> just pick any k indices
            rng.shuffle(indexes)
            return indexes[:k]

        A = np.asarray(sub, dtype=np.float32)  # (n,d)
        # if any zero rows (all zeros), keep them as zeros (already normalized to [])
        # choose a random start
        first_pos = rng.randrange(len(indexes))
        chosen_pos = [first_pos]
        chosen = [indexes[first_pos]]

        # min distance to the chosen set for every candidate
        min_d = _cosine_dist_vec(A, A[first_pos])  # (n,)
        min_d[first_pos] = -1.0  # exclude itself

        if progress:
            progress(1, k)

        while len(chosen) < k and len(chosen) < len(indexes):
            # pick farthest from current chosen set
            nxt_pos = int(min_d.argmax())
            chosen.append(indexes[nxt_pos])
            chosen_pos.append(nxt_pos)

            # update min_d with distances to the new chosen point
            d_new = _cosine_dist_vec(A, A[nxt_pos])
            # elementwise min
            np.minimum(min_d, d_new, out=min_d)
            min_d[nxt_pos] = -1.0  # exclude picked

            if progress:
                progress(len(chosen), k)

        return chosen

    # ------- Fallback: pure-Python O(k·n) -------
    # Pre-normalize only the needed vectors
    sub = [normalize_unit(vecs[i]) for i in indexes]
    if not sub or not sub[0]:
        rng.shuffle(indexes)
        return indexes[:k]

    # pick random start
    first_pos = rng.randrange(len(indexes))
    chosen_pos = [first_pos]
    chosen = [indexes[first_pos]]

    # initialize min distances to “distance to first”
    def _cos_dist(u, v):
        return 1.0 - cos_safe(u, v)

    min_d = [_cos_dist(sub[i], sub[first_pos]) for i in range(len(indexes))]
    min_d[first_pos] = -1.0  # exclude itself

    if progress:
        progress(1, k)

    while len(chosen) < k and len(chosen) < len(indexes):
        # farthest
        nxt_pos, best_d = -1, -1.0
        for i, d in enumerate(min_d):
            if d > best_d:
                best_d, nxt_pos = d, i
        chosen.append(indexes[nxt_pos])
        chosen_pos.append(nxt_pos)

        # update min distances w.r.t. new chosen point
        vi = sub[nxt_pos]
        for i in range(len(min_d)):
            if min_d[i] < 0.0:  # already chosen
                continue
            d_new = _cos_dist(sub[i], vi)
            if d_new < min_d[i]:
                min_d[i] = d_new
        min_d[nxt_pos] = -1.0

        if progress:
            progress(len(chosen), k)

    return chosen
