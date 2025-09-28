# stephanie/utils/statistics_utils.py
from __future__ import annotations

import math
import random
from typing import List, Tuple

def winsorize(xs: List[float], alpha: float = 0.05) -> List[float]:
    """Winsorize to reduce outlier impact (clamp tails)."""
    if not xs:
        return xs
    x = sorted(xs) 
    n = len(x)
    k = max(0, min(n - 1, int(alpha * n)))
    lo = x[k]
    hi = x[-k - 1] if k < n else x[-1]
    return [min(max(v, lo), hi) for v in x]

def mean_stdev(xs: List[float]) -> Tuple[float, float]:
    """Unbiased SD (n-1 in variance)."""
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(max(v, 0.0))

def welch_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Welch's t-test; returns (t_stat, p_two_sided) with normal approx."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    ma, sa = mean_stdev(a)
    mb, sb = mean_stdev(b)
    sa2, sb2 = sa ** 2, sb ** 2
    denom = math.sqrt((sa2 / na) + (sb2 / nb)) or 1e-9
    t = (mb - ma) / denom
    # Normal approximation for p-value
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
    return t, p

def mann_whitney_u(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Mann–Whitney U test with normal approximation and continuity correction.
    Returns (u_stat, p_two_sided).
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0, 1.0

    # Rank all values (average ranks for ties)
    vals = sorted([(v, 0) for v in a] + [(v, 1) for v in b], key=lambda t: t[0])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j][0] == vals[i][0]:
            j += 1
        r = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = r
        i = j

    ra = sum(ranks[i] for i in range(len(vals)) if vals[i][1] == 0)
    rb = sum(ranks[i] for i in range(len(vals)) if vals[i][1] == 1)
    ua = ra - na * (na + 1) / 2.0
    ub = rb - nb * (nb + 1) / 2.0
    u = min(ua, ub)

    mu = na * nb / 2.0
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12.0) or 1e-9
    z = (u - mu + 0.5) / sigma  # continuity correction
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))
    return u, p

def bootstrap_ci(a: List[float], b: List[float], iters: int = 2000, seed: int = 42) -> Tuple[float, float, float, float]:
    """
    Non-parametric bootstrap CI for:
      - delta = mean(B) - mean(A)
      - relative improvement = delta / |mean(A)|
    Returns (delta_lo, delta_hi, rel_lo, rel_hi) as 95% CIs.
    """
    if not a or not b:
        return 0.0, 0.0, 0.0, 0.0
    rnd = random.Random(seed)
    deltas, rels = [], []
    na, nb = len(a), len(b)
    for _ in range(iters):
        sa = [a[rnd.randrange(na)] for _ in range(na)]
        sb = [b[rnd.randrange(nb)] for _ in range(nb)]
        ma = sum(sa) / na
        mb = sum(sb) / nb
        d = mb - ma
        r = (d / abs(ma)) if ma != 0 else 0.0
        deltas.append(d)
        rels.append(r)
    deltas.sort()
    rels.sort()
    lo_idx = int(0.025 * len(deltas))
    hi_idx = max(0, int(0.975 * len(deltas)) - 1)
    return deltas[lo_idx], deltas[hi_idx], rels[lo_idx], rels[hi_idx]

def cohens_d(a: List[float], b: List[float]) -> float:
    """Pooled-SD Cohen's d; safe for small samples (returns 0 if degenerate)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, sa = mean_stdev(a)
    mb, sb = mean_stdev(b)
    sp_num = ((len(a) - 1) * (sa ** 2) + (len(b) - 1) * (sb ** 2))
    sp_den = (len(a) + len(b) - 2)
    if sp_den <= 0 or sp_num <= 0:
        return 0.0
    sp = math.sqrt(sp_num / sp_den)
    if sp == 0:
        return 0.0
    return (mb - ma) / sp


