# stephanie/utils/graph_utils.py
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# --------- small utilities ---------
def _percentiles(vals: Sequence[float], ps=(0.10, 0.25, 0.50, 0.75, 0.90)) -> Dict[str, float]:
    if not vals:
        return {f"p{int(p*100)}": 0.0 for p in ps}
    xs = sorted(float(x) for x in vals)
    n = len(xs)
    out: Dict[str, float] = {}
    for p in ps:
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        out[f"p{int(p*100)}"] = xs[i]
    return out


def _degree_list(adj: Dict[str, set]) -> List[int]:
    return [len(nbrs) for nbrs in adj.values()]


def _gini(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return 0.0
    xs.sort()
    mean = sum(xs) / n or 1.0
    num = 0.0
    for i, x in enumerate(xs, 1):
        num += (2 * i - n - 1) * x
    return num / (n * n * mean)


def _edge_fields(e: Any) -> Tuple[str, str, float, str]:
    """Robustly extract (src, dst, weight, type) from edge object or dict."""
    src = str(getattr(e, "src", getattr(e, "source", "")) or (e.get("src") if isinstance(e, dict) else ""))  # type: ignore
    dst = str(getattr(e, "dst", getattr(e, "target", "")) or (e.get("dst") if isinstance(e, dict) else ""))  # type: ignore
    w = getattr(e, "weight", None)
    if w is None and isinstance(e, dict):
        w = e.get("weight", 0.0)
    try:
        w = float(w or 0.0)
    except Exception:
        w = 0.0
    et = str(getattr(e, "type", (e.get("type") if isinstance(e, dict) else "")) or "").lower()  # type: ignore
    return src, dst, w, et


def unique_undirected_edges(edges: Iterable[Any]) -> List[Tuple[str, str, float, str]]:
    """Return undirected edges as (u,v,w,etype) with u<v and max weight kept if duplicated."""
    seen = {}
    for e in edges:
        u, v, w, et = _edge_fields(e)
        if not u or not v or u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key not in seen or w > seen[key][0]:
            seen[key] = (w, et)
    out = [(a, b, w, et) for (a, b), (w, et) in seen.items()]
    return out


def edge_type_counts(undirected_edges: Sequence[Tuple[str, str, float, str]]) -> Dict[str, int]:
    c = {"knn": 0, "temporal": 0, "other": 0}
    for _, _, _, et in undirected_edges:
        if "knn" in et:
            c["knn"] += 1
        elif "temp" in et or "backtrack" in et or "time" in et:
            c["temporal"] += 1
        else:
            c["other"] += 1
    return c


# --------- core graph primitives ---------
def build_adjacency(edges: Iterable[Any]) -> Tuple[Dict[str, set], Dict[Tuple[str, str], float]]:
    adj: Dict[str, set] = defaultdict(set)
    ew: Dict[Tuple[str, str], float] = {}
    for e in edges:
        src, dst, w, _ = _edge_fields(e)
        if not src or not dst or src == dst:
            continue
        adj[src].add(dst)
        adj[dst].add(src)
        ew[(src, dst)] = w
        ew[(dst, src)] = w
    return adj, ew


def connected_components(adj: Dict[str, set]) -> List[List[str]]:
    seen = set()
    comps: List[List[str]] = []
    for u in list(adj.keys()):
        if u in seen:
            continue
        stack = [u]
        seen.add(u)
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(comp)
    return comps


def approx_clustering_coefficient(adj: Dict[str, set]) -> float:
    N = 0
    S = 0.0
    for _, nbrs in adj.items():
        k = len(nbrs)
        if k < 2:
            continue
        nn = 0
        nbrs_list = list(nbrs)
        L = len(nbrs_list)
        for i in range(L):
            a = nbrs_list[i]
            for j in range(i + 1, L):
                b = nbrs_list[j]
                if a in adj[b]:
                    nn += 1
        possible = k * (k - 1) / 2
        S += nn / possible
        N += 1
    return (S / N) if N else 0.0


def mutual_knn_fraction(edges: Iterable[Any]) -> float:
    knn = set()
    for e in edges:
        src, dst, _, et = _edge_fields(e)
        if not src or not dst or src == dst:
            continue
        if "knn" in et:
            knn.add((src, dst))
    if not knn:
        return 0.0
    mutual = sum(1 for (a, b) in knn if (b, a) in knn)
    return mutual / len(knn)


def degree_assortativity(adj: Dict[str, set], undirected_edges: Sequence[Tuple[str, str, float, str]]) -> float:
    if not undirected_edges:
        return 0.0
    deg = {u: len(adj.get(u, ())) for u in adj}
    xs, ys = [], []
    for u, v, _, _ in undirected_edges:
        xs.append(deg.get(u, 0))
        ys.append(deg.get(v, 0))
    n = len(xs)
    if n == 0:
        return 0.0
    meanx = sum(xs) / n
    meany = sum(ys) / n
    num = sum((x - meanx) * (y - meany) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - meanx) ** 2 for x in xs) or 1.0)
    deny = math.sqrt(sum((y - meany) ** 2 for y in ys) or 1.0)
    return num / (denx * deny or 1.0)


def kcore_stats(adj: Dict[str, set]) -> Dict[str, int]:
    """Return largest k-core (kmax) and its size via simple peeling."""
    from collections import deque
    deg = {u: len(nbrs) for u, nbrs in adj.items()}
    nbrs = {u: set(vs) for u, vs in adj.items()}
    remaining = set(adj.keys())
    kmax = 0

    # Seed peel <1
    queue = deque([u for u, d in deg.items() if d < 1])
    while queue:
        u = queue.popleft()
        if u not in remaining:
            continue
        remaining.remove(u)
        for v in list(nbrs[u]):
            if v in remaining:
                nbrs[v].discard(u)
                deg[v] -= 1
                if deg[v] < 1:
                    queue.append(v)

    # Raise k while possible
    while remaining:
        next_k = kmax + 1
        peeled = True
        while peeled:
            peeled = False
            for u in list(remaining):
                if deg[u] < next_k:
                    remaining.remove(u)
                    for v in list(nbrs[u]):
                        if v in remaining:
                            nbrs[v].discard(u)
                            deg[v] -= 1
                    peeled = True
        if not remaining:
            break
        kmax = next_k
        # loop continues until no more raise is possible
        if kmax > len(adj):
            break

    return {"kmax": int(kmax), "kmax_core_size": int(len(remaining))}


def label_prop_communities(adj: Dict[str, set], max_iters: int = 10, seed: int = 0) -> List[set]:
    """Quick label-prop, deterministic tie-break by label, fixed node ordering."""
    import random
    rng = random.Random(seed)  # noqa: F841 (kept for future randomized variants)
    labels = {u: u for u in adj}
    nodes = list(adj.keys())
    for _ in range(max_iters):
        changed = 0
        for u in nodes:
            if not adj[u]:
                continue
            counts: Dict[Any, int] = {}
            for v in adj[u]:
                lv = labels[v]
                counts[lv] = counts.get(lv, 0) + 1
            best = sorted(counts.items(), key=lambda t: (-t[1], str(t[0])))[0][0]
            if labels[u] != best:
                labels[u] = best
                changed += 1
        if changed == 0:
            break
    parts: Dict[Any, set] = {}
    for u, lab in labels.items():
        parts.setdefault(lab, set()).add(u)
    return list(parts.values())


def weighted_modularity(parts: Sequence[set], undirected_edges: Sequence[Tuple[str, str, float, str]]) -> float:
    """Newman–Girvan modularity for weighted, undirected graphs."""
    if not undirected_edges:
        return 0.0
    strength: Dict[str, float] = {}
    m = 0.0
    for u, v, w, _ in undirected_edges:
        m += w
        strength[u] = strength.get(u, 0.0) + w
        strength[v] = strength.get(v, 0.0) + w
    if m <= 0:
        return 0.0
    two_m = 2.0 * m
    nid: Dict[str, int] = {}
    for i, part in enumerate(parts):
        for u in part:
            nid[u] = i
    sum_in = [0.0 for _ in parts]
    sum_tot = [0.0 for _ in parts]
    for i, part in enumerate(parts):
        sum_tot[i] = sum(strength.get(u, 0.0) for u in part)
    for u, v, w, _ in undirected_edges:
        if nid.get(u) == nid.get(v):
            sum_in[nid[u]] += w
    Q = 0.0
    for i in range(len(parts)):
        Q += (sum_in[i] / two_m) - (sum_tot[i] / two_m) ** 2
    return float(Q)


# --------- metrics beyond pure topology (alignment, spatial, aggregations) ---------
def goal_alignment_stats(manifest, target_vec: Sequence[float], restrict_to_ids: Optional[set[str]] = None) -> Dict[str, float]:
    """Cosine(sim(item, goal)), robust to missing vectors. Requires manifest.items with embeddings/metrics_vector."""
    def _l2(v):
        return math.sqrt(sum(x * x for x in v)) or 1.0

    def _cos(a, b):
        if not a or not b:
            return 0.0
        a = list(a); b = list(b)
        return sum(x * y for x, y in zip(a, b)) / (_l2(a) * _l2(b))

    sims: List[float] = []
    for mi in manifest.items:
        if restrict_to_ids and mi.item_id not in restrict_to_ids:
            continue
        v = (mi.embeddings or {}).get("global")
        if v and isinstance(v, (list, tuple)):
            sims.append(_cos(v, target_vec))
        elif mi.metrics_vector:
            sims.append(_cos(list(mi.metrics_vector.values()), target_vec))
        else:
            sims.append(0.0)
    if not sims:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "count": 0}
    s = sorted(float(x) for x in sims)
    n = len(s)
    def pct(p):
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return s[i]
    return {"mean": sum(s) / n, "median": s[n // 2], "p90": pct(0.90), "count": n}


def aggregate_metric_columns(manifest) -> Dict[str, Dict[str, float]]:
    """Aggregate per-item (metrics_columns, metrics_values) → {col: {mean,p90,count}}."""
    from collections import defaultdict
    buckets: Dict[str, List[float]] = defaultdict(list)
    for mi in manifest.items:
        cols = list(mi.metrics_columns or [])
        vals = list(mi.metrics_values or [])
        for k, v in zip(cols, vals):
            try:
                buckets[str(k)].append(float(v))
            except Exception:
                continue
    out: Dict[str, Dict[str, float]] = {}
    for k, arr in buckets.items():
        if not arr:
            continue
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        p90 = arr_sorted[min(n - 1, max(0, int(round(0.9 * (n - 1)))))]
        out[k] = {"mean": float(sum(arr_sorted) / n), "p90": float(p90), "count": int(n)}
    return out


# --------- top-level run metrics aggregator ---------
def compute_run_metrics(
    *,
    manifest,
    nodes: Dict[str, Any],
    edges: Sequence[Any],
    positions: Dict[str, Tuple[float, float]],
    target_vec: Sequence[float],
    params: Optional[dict] = None,
) -> Dict[str, Any]:
    adj, _ = build_adjacency(edges)
    undirected = unique_undirected_edges(edges)

    # counts
    n_nodes = len(nodes)
    n_edges = len(undirected)
    avg_deg = (2.0 * n_edges / n_nodes) if n_nodes else 0.0

    # components
    comps = connected_components(adj)
    comp_sizes = [len(c) for c in comps]
    largest_cc = max(comp_sizes) if comp_sizes else 0
    giant_ratio = (largest_cc / n_nodes) if n_nodes else 0.0

    # degree stats
    degs = _degree_list(adj)
    deg_mean = float(sum(degs) / len(degs)) if degs else 0.0
    deg_var = float(sum((d - deg_mean) ** 2 for d in degs) / len(degs)) if degs else 0.0
    degree_stats = {
        "mean": deg_mean,
        "std": deg_var ** 0.5 if deg_var > 0 else 0.0,
        **_percentiles(degs, (0.5, 0.9)),
        "gini": _gini(degs),
        "assortativity": degree_assortativity(adj, undirected),
    }
    kcore = kcore_stats(adj)

    # edge weights
    wts = [w for _, _, w, _ in undirected]
    wt_mean = float(sum(wts) / len(wts)) if wts else 0.0
    wt_var = float(sum((w - wt_mean) ** 2 for w in wts) / len(wts)) if wts else 0.0
    wt_stats = {"mean": wt_mean, "std": wt_var ** 0.5 if wt_var > 0 else 0.0, **_percentiles(wts, (0.5, 0.9))}

    # spatial geometry
    lens = []
    for u, v, _, _ in undirected:
        if (u in positions) and (v in positions):
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            if None not in (x1, y1, x2, y2):
                dx, dy = (x1 - x2), (y1 - y2)
                lens.append((dx * dx + dy * dy) ** 0.5)
    spatial = {"mean_edge_len": 0.0, "long_edge_frac": 0.0}
    if lens:
        pct = _percentiles(lens, (0.10, 0.25, 0.50, 0.75, 0.90))
        mean_len = float(sum(lens) / len(lens))
        p75 = pct.get("p75", 0.0)
        long_frac = float(sum(1 for L in lens if L >= p75) / len(lens)) if len(lens) else 0.0
        spatial = {"mean_edge_len": mean_len, **pct, "long_edge_frac": long_frac}

    # clustering + reciprocity + types
    cluster_c = approx_clustering_coefficient(adj)
    mutual_knn = mutual_knn_fraction(edges)
    edge_mix = edge_type_counts(undirected)

    # communities + modularity
    parts = label_prop_communities(adj, max_iters=10, seed=0) if n_nodes <= 5000 else []
    modularity = weighted_modularity(parts, undirected) if parts else 0.0
    community = {
        "method": "label_prop",
        "count": len(parts),
        "size_p90": _percentiles([len(p) for p in parts], (0.9,)).get("p90", 0.0) if parts else 0.0,
        "modularity": modularity,
    }

    # alignment + scorer aggregates
    align = goal_alignment_stats(manifest, target_vec)
    metric_cols = aggregate_metric_columns(manifest)

    provenance = dict(params or {})

    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "avg_degree": avg_deg,
        "degree": degree_stats,
        "kcore": kcore,
        "components": {
            "count": len(comps),
            "sizes": comp_sizes[:50],
            "giant_ratio": giant_ratio,
            "largest_component": largest_cc,
        },
        "edge_weights": wt_stats,
        "edge_types": edge_mix,
        "clustering_coeff": cluster_c,
        "mutual_knn_frac": mutual_knn,
        "spatial": spatial,
        "community": community,
        "goal_alignment": align,
        "metric_columns": metric_cols,
        "provenance": provenance,
    }

