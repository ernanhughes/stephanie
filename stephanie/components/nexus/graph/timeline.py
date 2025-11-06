from __future__ import annotations
from typing import Dict, List, Any, Tuple, Callable, Optional, Iterable
import json
from contextlib import contextmanager

ProgressFn = Callable[[int, int], None]  # (done_steps, total_steps) -> None

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
    """Light wrapper so tqdm is optional and zero-overhead if missing/disabled."""
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
) -> List[Dict[str, Any]]:
    """
    Progressive timeline: step k shows all nodes with item_index <= k and all
    edges whose endpoints both appear by step k.

    Args:
      progress: optional callback (done, total)
      tqdm_progress: set True to show a tqdm bar (if tqdm installed)
      stream_to_path: if set, write JSON array directly to this path and return [].
                      (Much faster + lower memory for large graphs.)
    """
    # 1) Map item -> step
    item_index = {getattr(mi, "item_id", None) or getattr(mi, "id", None): i
                  for i, mi in enumerate(manifest_items)}
    def _idx(nid: str) -> int:
        return item_index.get(nid, 10**9)

    # 2) Convert nodes once and bucket by step
    cy_nodes_all = _to_cyto_nodes(nodes, positions)
    node_step = {n["data"]["id"]: _idx(n["data"]["id"]) for n in cy_nodes_all}

    # 3) Compute edge step once, convert once, and bucket by step
    edge_buckets: Dict[int, List[Dict[str, Any]]] = {}
    max_step = 0
    for e in edges:
        s = getattr(e, "src", None) or getattr(e, "source", None)
        t = getattr(e, "dst", None) or getattr(e, "target", None)
        st = max(_idx(str(s)), _idx(str(t)))
        cy_e = _edge_to_cyto(e)
        edge_buckets.setdefault(st, []).append(cy_e)
        if st > max_step:
            max_step = st

    # sort each bucket by weight desc for nicer reveal
    for st, lst in edge_buckets.items():
        lst.sort(key=lambda ce: -(ce["data"].get("weight") or 0.0))

    total_steps = max(max_step, *(item_index.values() or [0])) + 1

    # Prepare node buckets so we only append newcomers each step
    node_buckets: Dict[int, List[Dict[str, Any]]] = {}
    for n in cy_nodes_all:
        st = node_step.get(n["data"]["id"], 10**9)
        node_buckets.setdefault(st, []).append(n)

    # 4) Accumulate without reconversion
    frames_acc_nodes: List[Dict[str, Any]] = []
    frames_acc_edges: List[Dict[str, Any]] = []

    # Optional streaming writer
    writer = None
    if stream_to_path:
        writer = open(stream_to_path, "w", encoding="utf-8")
        writer.write("[")

    frames_out: List[Dict[str, Any]] = []
    sep = ""  # for streaming commas

    with _maybe_tqdm(total_steps, tqdm_progress) as tick:
        for k in range(total_steps):
            # append newly available nodes/edges for this step
            frames_acc_nodes.extend(node_buckets.get(k, ()))
            frames_acc_edges.extend(edge_buckets.get(k, ()))

            frame = {
                "nodes": frames_acc_nodes,   # NOTE: we copy below to freeze per-frame
                "edges": frames_acc_edges,
            }

            if writer:
                # Write a compact copy to disk (avoid mutating previous frames)
                writer.write(sep)
                json.dump({
                    "nodes": frames_acc_nodes.copy(),
                    "edges": frames_acc_edges.copy(),
                }, writer, ensure_ascii=False)
                sep = ","
            else:
                # Keep in memory: copy the lists so later appends don't mutate older frames
                frames_out.append({
                    "nodes": frames_acc_nodes.copy(),
                    "edges": frames_acc_edges.copy(),
                })

            if progress:
                progress(k + 1, total_steps)
            tick(1)

    if writer:
        writer.write("]")
        writer.close()
        return []  # streamed to disk

    # guard
    if not frames_out:
        frames_out = [{"nodes": cy_nodes_all, "edges": sum(edge_buckets.values(), [])}]
    return frames_out
