from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

ProgressFn = Callable[[int, int], None]

def _to_cyto_nodes(nodes: Dict[str, Any],
                   positions: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
    out = []
    for nid, n in nodes.items():
        x, y = positions.get(nid, (None, None))
        out.append({
            "data": {
                "id": nid,
                "label": getattr(n, "title", None) or getattr(n, "text", "")[:80] or nid,
                "type": getattr(n, "target_type", "node"),
                "weight": float(getattr(n, "weight", 0.0) or 0.0),
            },
            "position": {"x": x, "y": y} if x is not None and y is not None else None,
        })
    return out

def _edge_to_cyto(e: Any) -> Dict[str, Any]:
    src = getattr(e, "src", None) or getattr(e, "source", None)
    dst = getattr(e, "dst", None) or getattr(e, "target", None)
    ety = getattr(e, "type", "edge")
    w   = float(getattr(e, "weight", 0.0) or 0.0)
    return {"data": {"id": f"{src}->{dst}", "source": src, "target": dst, "type": ety, "weight": w}}

@contextmanager
def _maybe_tqdm(total: int, enable: bool):
    pbar = None
    if enable:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total, leave=False, desc="frames")
        except Exception:
            pbar = None
    try:
        yield (lambda n: pbar.update(n) if pbar else None)
    finally:
        if pbar:
            pbar.close()

def build_frames_from_manifest(
    manifest_items: List[Any],
    nodes: Dict[str, Any],
    edges: List[Any],
    positions: Dict[str, Tuple[float, float]],
    *,
    progress: Optional[ProgressFn] = None,
    tqdm_progress: bool = False,
    stream_to_path: Optional[str] = None,
    drop_orphan_edges: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build progressive frames; compact step indices so the first non-empty step is 0.
    Nodes that don't match any manifest item fall back to step 0 to avoid leading empties.
    """
    # item -> step (by manifest order)
    item_index = {getattr(mi, "item_id", None) or getattr(mi, "id", None): i
                  for i, mi in enumerate(manifest_items)}

    # Convert nodes once
    cy_nodes_all = _to_cyto_nodes(nodes, positions)

    # 1) Assign each node an initial step:
    #    - exact id match -> that manifest step
    #    - else fallback to 0 (prevents huge leading empty frames)
    node_step_raw: Dict[str, int] = {}
    for n in cy_nodes_all:
        nid = n["data"]["id"]
        node_step_raw[nid] = item_index.get(nid, 0)

    # 2) Compact steps: map sorted unique steps to 0..N-1
    unique_steps = sorted(set(node_step_raw.values()))
    if not unique_steps:
        unique_steps = [0]
    step_map = {s: i for i, s in enumerate(unique_steps)}

    node_step: Dict[str, int] = {nid: step_map[s] for nid, s in node_step_raw.items()}

    # Bucket nodes by compacted step
    node_buckets: Dict[int, List[Dict[str, Any]]] = {}
    for n in cy_nodes_all:
        st = node_step[n["data"]["id"]]
        node_buckets.setdefault(st, []).append(n)

    # 3) Edges: keep only those with known endpoints; step = max(node steps)
    node_ids = set(nodes.keys())
    edge_buckets: Dict[int, List[Dict[str, Any]]] = {}
    for e in edges:
        s = getattr(e, "src", None) or getattr(e, "source", None)
        t = getattr(e, "dst", None) or getattr(e, "target", None)
        if s is None or t is None:
            continue
        if drop_orphan_edges and (s not in node_ids or t not in node_ids):
            continue
        st = max(node_step.get(str(s), 0), node_step.get(str(t), 0))
        edge_buckets.setdefault(st, []).append(_edge_to_cyto(e))

    # (optional) local sort by weight for nicer reveal
    for st, lst in edge_buckets.items():
        lst.sort(key=lambda ce: -(ce["data"].get("weight") or 0.0))

    # 4) Total steps from what actually exists
    all_steps = set(node_buckets.keys()) | set(edge_buckets.keys())
    total_steps = (max(all_steps) + 1) if all_steps else 1

    # 5) Accumulate
    acc_nodes: List[Dict[str, Any]] = []
    acc_edges: List[Dict[str, Any]] = []

    writer = None
    if stream_to_path:
        writer = open(stream_to_path, "w", encoding="utf-8")
        writer.write("[")

    out: List[Dict[str, Any]] = []
    sep = ""

    with _maybe_tqdm(total_steps, tqdm_progress) as tick:
        for k in range(total_steps):
            acc_nodes.extend(node_buckets.get(k, ()))
            acc_edges.extend(edge_buckets.get(k, ()))

            frame = {"nodes": acc_nodes.copy(), "edges": acc_edges.copy()}
            if writer:
                writer.write(sep); json.dump(frame, writer, ensure_ascii=False); sep = ","
            else:
                out.append(frame)

            if progress: progress(k + 1, total_steps)
            tick(1)

    if writer:
        writer.write("]"); writer.close(); return []

    if not out:
        all_edges = [e for lst in edge_buckets.values() for e in lst]
        out = [{"nodes": cy_nodes_all, "edges": all_edges}]
    return out
