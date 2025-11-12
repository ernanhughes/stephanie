# stephanie/components/nexus/ab_planner.py
from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _l2(v):
    return math.sqrt(sum(x * x for x in v)) or 1.0


def _cos(a: Iterable[float] | None, b: Iterable[float] | None) -> float:
    if not a or not b:
        return 0.0
    a = list(a)
    b = list(b)
    return sum(x * y for x, y in zip(a, b)) / (_l2(a) * _l2(b))


def _get_vec(s: Dict[str, Any]) -> List[float]:
    em = (s.get("embeddings") or {}).get("global")
    if em:
        return em
    vec = s.get("metrics_vector") or {}
    if vec:
        return list(vec.values())
    return []  # caller can backfill using ZeroModel if desired


def _topk_neighbors(
    items: List[Dict[str, Any]],
    idx: int,
    goal_vec: List[float],
    k: int,
    alpha: float,
) -> Tuple[List[int], float]:
    """
    For item i, pick k neighbors that jointly maximize a bridge score.
    Bridge score for neighbor j: sqrt(sim(i,j) * sim(j,goal)).
    Item score = alpha*sim(i,goal) + (1-alpha)*max_j bridge(i->j->goal).
    """
    vi = _get_vec(items[idx])
    s_i_g = _cos(vi, goal_vec)
    cand = []
    for j, sj in enumerate(items):
        if j == idx:
            continue
        vj = _get_vec(sj)
        s_ij = _cos(vi, vj)
        s_jg = _cos(vj, goal_vec)
        bridge = math.sqrt(max(0.0, s_ij) * max(0.0, s_jg))
        cand.append((bridge, j, s_ij, s_jg))
    cand.sort(reverse=True, key=lambda t: t[0])
    chosen = [j for _, j, _, _ in cand[: max(0, k)]]
    best_bridge = cand[0][0] if cand else 0.0
    score = alpha * s_i_g + (1.0 - alpha) * best_bridge
    return chosen, score


def plan_ab_runs(
    scorables: List[Dict[str, Any]],
    *,
    goal_vec: List[float],
    neighbors_per_item: int = 4,
    alpha_direct: float = 0.6,  # weight for direct sim(item, goal)
    seed: int = 0,
    max_items: Optional[int] = None,  # cap for demo
) -> Dict[str, Any]:
    """
    Returns:
      baseline_ids: indices in original order (possibly capped)
      targeted_ids: indices ordered by goal-bridge score (deduped)
      neighbor_map: {i: [j1..jk]} for display/context
      scores: {i: float} overall score used for ordering
    """
    rng = random.Random(seed)
    n = len(scorables)
    order = list(range(n))
    if max_items is not None:
        order = order[:max_items]

    # Baseline: original order (or timestamp order if present)
    baseline_ids = order[:]

    # Per-item neighbor set + score
    neighbor_map: Dict[int, List[int]] = {}
    scores: Dict[int, float] = {}
    for i in order:
        neigh, s = _topk_neighbors(
            scorables, i, goal_vec, neighbors_per_item, alpha_direct
        )
        neighbor_map[i] = neigh
        scores[i] = s

    # Targeted: sort by score desc, but avoid flooding with duplicates by interleaving neighbors
    primary = sorted(order, key=lambda i: -(scores.get(i, 0.0)))
    targeted_ids: List[int] = []
    seen = set()
    for i in primary:
        if i not in seen:
            targeted_ids.append(i)
            seen.add(i)
        for j in neighbor_map.get(i, []):
            if j not in seen:
                targeted_ids.append(j)
                seen.add(j)

    # truncate to same length as baseline for a clean A/B
    if len(targeted_ids) > len(baseline_ids):
        targeted_ids = targeted_ids[: len(baseline_ids)]

    return {
        "baseline_ids": baseline_ids,
        "targeted_ids": targeted_ids,
        "neighbor_map": neighbor_map,
        "scores": scores,
    }
