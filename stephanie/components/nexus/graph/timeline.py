# stephanie/components/nexus/graph/timeline.py
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

ProgressFn = Callable[[int, int], None]


# ---------------------------
# Small helpers
# ---------------------------

def _sid(x: Any) -> str:
    """Stable string id."""
    return str(x) if x is not None else ""

def _get(obj: Any, *keys, default=None):
    for k in keys:
        if hasattr(obj, k):
            return getattr(obj, k)
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return default


# ---------------------------
# Cytoscape converters
# ---------------------------

def _node_label(n: Any) -> str:
    # Prefer human label: title -> text -> sharpened/state text
    title = _get(n, "title", "label")
    if title:
        return str(title)[:80]
    txt = _get(n, "text", "sharpened_text", "state_text")
    if txt:
        return str(txt).strip().replace("\n", " ")[:80]
    return _sid(_get(n, "id"))

def _node_role(n: Any) -> str:
    # winner|candidate|root|baseline|node
    tags = _get(n, "tags") or []
    if isinstance(tags, str):
        try:
            import json as _json
            tags = _json.loads(tags)
        except Exception:
            tags = [tags]
    tags = [str(t).lower() for t in (tags or [])]
    if "winner" in tags:
        return "winner"
    if "candidate" in tags:
        return "candidate"
    if "root" in tags:
        return "root"
    if "baseline" in tags:
        return "baseline"
    return "node"

def _node_weight(n: Any) -> float:
    # prefer scores.overall, else weight attr
    scores = _get(n, "scores") or {}
    try:
        return float(scores.get("overall", _get(n, "weight", default=0.0)) or 0.0)
    except Exception:
        return 0.0

def _to_cyto_nodes(nodes: Dict[str, Any],
                   positions: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, Any]]:
    """
    Return dict id->cyto node so we can add classes later without copying.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for nid, n in nodes.items():
        nid_s = _sid(nid)
        x, y = positions.get(nid_s, (None, None))
        out[nid_s] = {
            "data": {
                "id": nid_s,
                "label": _node_label(n),
                "type": _get(n, "target_type", default="node"),
                "role": _node_role(n),
                "accepted": bool(_get(n, "accepted", default=False)),
                "weight": _node_weight(n),
            },
            # We'll add "classes" per-frame
            "classes": "",
            "position": {"x": x, "y": y} if (x is not None and y is not None) else None,
        }
    return out

def _edge_to_cyto(e: Any) -> Dict[str, Any]:
    src = _sid(_get(e, "src", "source", "src_node_id"))
    dst = _sid(_get(e, "dst", "target", "dst_node_id"))
    ety = _get(e, "type", "relation", default="edge")
    w   = float(_get(e, "weight", "score", default=0.0) or 0.0)
    return {"data": {"id": f"{src}->{dst}", "source": src, "target": dst, "type": ety, "weight": w}, "classes": ""}


# ---------------------------
# Progress / streaming
# ---------------------------

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


# ---------------------------
# Manifest parsing
# ---------------------------

class _ManifestIndex:
    """
    Interprets a list of 'events' that may come from:
      - Blossom episode logs
      - Runner events
      - Ad-hoc dicts

    Recognized fields (best-effort):
      event/kind: 'add_node'|'node'|'winner'|'reject'|'edge_add'|'edge'|'remove_edge'
      bn_id/node_id
      src/src_node_id/source, dst/dst_node_id/target
    """
    def __init__(self, items: List[Any]):
        self.node_create_step: Dict[str, int] = {}
        self.node_reject_step: Dict[str, int] = {}
        self.node_winner_step: Dict[str, int] = {}
        self.edge_create_step: Dict[Tuple[str, str], int] = {}
        self.edge_remove_step: Dict[Tuple[str, str], int] = {}

        for i, mi in enumerate(items or []):
            evt = (_get(mi, "event", "kind", default="") or "").lower()
            nid = _sid(_get(mi, "bn_id", "node_id", "id"))
            src = _sid(_get(mi, "src", "src_node_id", "source"))
            dst = _sid(_get(mi, "dst", "dst_node_id", "target"))

            if nid:
                if evt in ("add_node", "node", "new_node") and nid not in self.node_create_step:
                    self.node_create_step[nid] = i
                elif evt in ("reject", "prune", "drop") and nid not in self.node_reject_step:
                    self.node_reject_step[nid] = i
                elif evt in ("winner", "select"):
                    self.node_winner_step.setdefault(nid, i)

            if src and dst:
                key = (src, dst)
                if evt in ("edge_add", "edge", "link"):
                    self.edge_create_step.setdefault(key, i)
                elif evt in ("edge_remove", "unlink", "prune_edge") and key not in self.edge_remove_step:
                    self.edge_remove_step[key] = i


# ---------------------------
# Frame builder
# ---------------------------

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
    flash_new: int = 0,                  # duplicate a frame when new items appear
) -> List[Dict[str, Any]]:
    """
    Build progressive frames revealing nodes/edges in the order implied by `manifest_items`.

    Behavior with richer events:
      • A node appears at its 'create' step; disappears after its 'reject' step (if any).
      • A node gets 'winner' class at/after its winner step.
      • An edge appears at its 'create' step (or max(step(src), step(dst)) if unspecified);
        disappears after its 'remove' step (if any).

    If an id doesn't show in the manifest, it is assigned to step 0 (prevents empty lead-in).
    """
    mi = _ManifestIndex(manifest_items or [])

    # Convert once
    cy_nodes_all = _to_cyto_nodes(nodes, positions)
    node_ids = set(cy_nodes_all.keys())

    # ---- Determine per-node steps
    node_create: Dict[str, int] = {}
    node_reject: Dict[str, Optional[int]] = {}
    node_winner: Dict[str, Optional[int]] = {}

    for nid in node_ids:
        c_step = mi.node_create_step.get(nid, 0)
        r_step = mi.node_reject_step.get(nid)  # may be None
        w_step = mi.node_winner_step.get(nid)
        node_create[nid] = c_step
        node_reject[nid] = r_step
        node_winner[nid] = w_step

    # Compact step ids to 0..K-1 for stability
    uniq_steps = sorted({*node_create.values(), *mi.edge_create_step.values()} or {0})
    step_map = {s: i for i, s in enumerate(uniq_steps)}
    node_create = {nid: step_map[s] for nid, s in node_create.items()}
    node_reject = {nid: (step_map[r] if r is not None and r in step_map else r) for nid, r in node_reject.items()}
    node_winner = {nid: (step_map[w] if w is not None and w in step_map else w) for nid, w in node_winner.items()}

    # ---- Bucket nodes by create step
    node_buckets: Dict[int, List[str]] = {}
    for nid, st in node_create.items():
        node_buckets.setdefault(st, []).append(nid)

    # ---- Edges
    edge_create_buckets: Dict[int, List[Dict[str, Any]]] = {}
    edge_remove_step: Dict[str, Optional[int]] = {}

    for e in edges or []:
        s = _sid(_get(e, "src", "source", "src_node_id"))
        t = _sid(_get(e, "dst", "target", "dst_node_id"))
        if not s or not t:
            continue
        if drop_orphan_edges and (s not in node_ids or t not in node_ids):
            continue

        key = (s, t)
        create_s = mi.edge_create_step.get(key)
        if create_s is None:
            # default: after both endpoints exist
            create_s = max(node_create.get(s, 0), node_create.get(t, 0))
        create_st = step_map.get(create_s, 0)

        edge_create_buckets.setdefault(create_st, []).append(_edge_to_cyto(e))
        # optional remove
        rm = mi.edge_remove_step.get(key)
        edge_remove_step[f"{s}->{t}"] = (step_map[rm] if rm is not None and rm in step_map else rm)

    # Sort edges by weight per step for nicer reveal
    for lst in edge_create_buckets.values():
        lst.sort(key=lambda ce: -(ce["data"].get("weight") or 0.0))

    # ---- Total steps
    all_steps = set(node_buckets.keys()) | set(edge_create_buckets.keys())
    total_steps = (max(all_steps) + 1) if all_steps else 1

    # ---- Accumulation
    active_nodes: Dict[str, Dict[str, Any]] = {}
    active_edges: Dict[str, Dict[str, Any]] = {}

    def _prune_for_step(k: int):
        # remove nodes rejected at/before k
        to_remove_nodes = [nid for nid, r in node_reject.items() if (r is not None and r <= k) and nid in active_nodes]
        for nid in to_remove_nodes:
            active_nodes.pop(nid, None)
            # drop connected edges
            for eid in [eid for eid in list(active_edges.keys()) if eid.startswith(f"{nid}->") or eid.endswith(f"->{nid}")]:
                active_edges.pop(eid, None)
        # remove edges with remove_step <= k
        for eid, rm in list(edge_remove_step.items()):
            if rm is not None and rm <= k and eid in active_edges:
                active_edges.pop(eid, None)

    writer = None
    if stream_to_path:
        writer = open(stream_to_path, "w", encoding="utf-8")
        writer.write("[")

    out: List[Dict[str, Any]] = []
    sep = ""

    def _emit_frame(new_node_ids: Iterable[str], new_edge_ids: Iterable[str], k: int):
        # Assign classes per-frame
        for nid, el in active_nodes.items():
            classes = []
            role = el["data"].get("role")
            accepted = bool(el["data"].get("accepted"))
            if nid in new_node_ids:
                classes.append("new")
                classes.append("active")
            if role:
                classes.append(role)  # winner/candidate/root/baseline/node
            if accepted:
                classes.append("accepted")
            if nid in node_winner and node_winner[nid] is not None and k >= node_winner[nid]:
                classes.append("winner")
            if node_reject.get(nid) is not None:
                # It's still here if reject step is in the future; mark as 'pending-reject'
                if k < node_reject[nid]:
                    classes.append("pending-reject")
                else:
                    classes.append("rejected")
            el["classes"] = " ".join(sorted(set(classes)))

        for eid, ee in active_edges.items():
            classes = []
            if eid in new_edge_ids:
                classes += ["new", "active"]
            ety = ee["data"].get("type")
            if ety:
                classes.append(str(ety))
            ee["classes"] = " ".join(sorted(set(classes)))

        frame = {
            "nodes": [active_nodes[nid] for nid in sorted(active_nodes.keys())],
            "edges": [active_edges[eid] for eid in sorted(active_edges.keys())],
            "meta": {"step": k}
        }
        if writer:
            nonlocal sep
            writer.write(sep)
            json.dump(frame, writer, ensure_ascii=False)
            sep = ","
        else:
            out.append(frame)

    with _maybe_tqdm(total_steps, tqdm_progress) as tick:
        for k in range(total_steps):
            # add nodes for this step
            just_nodes = []
            for nid in node_buckets.get(k, []):
                el = cy_nodes_all[nid]
                active_nodes[nid] = {"data": dict(el["data"]), "position": el["position"], "classes": ""}
                just_nodes.append(nid)

            # add edges for this step
            just_edges = []
            for ce in edge_create_buckets.get(k, []):
                eid = ce["data"]["id"]
                active_edges[eid] = {"data": dict(ce["data"]), "classes": ""}
                just_edges.append(eid)

            # remove anything that should disappear at/<= k
            _prune_for_step(k)

            # frame for this step
            _emit_frame(just_nodes, just_edges, k)

            # optional flash frames
            for _ in range(max(0, int(flash_new))):
                _emit_frame(just_nodes, just_edges, k)

            if progress:
                progress(k + 1, total_steps)
            tick(1)

    if writer:
        writer.write("]")
        writer.close()
        return []

    if not out:
        # Fallback single frame
        # Keep all nodes/edges, no classes
        all_edges = []
        for lst in edge_create_buckets.values():
            all_edges.extend(lst)
        out = [{"nodes": list(cy_nodes_all.values()), "edges": all_edges, "meta": {"step": 0}}]
    return out


# ---------------------------
# Blossom convenience
# ---------------------------

def frames_from_blossom(
    blossom: Any,
    *,
    positions: Dict[str, Tuple[float, float]],
    manifest_items: Optional[List[Any]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper to produce frames directly from a BlossomORM (or dict-like) object.
    If no manifest is provided, we synthesize one by node id order.
    """
    nodes = { _sid(_get(n, "id")): n for n in _get(blossom, "nodes", default=[]) }
    edges = list(_get(blossom, "edges", default=[]))

    if manifest_items is None:
        # Synth: add nodes in id order; mark winners/accept/reject from node fields/tags
        manifest_items = []
        for n in sorted(nodes.values(), key=lambda x: int(_get(x, "id", default=0))):
            nid = _sid(_get(n, "id"))
            manifest_items.append({"event": "add_node", "bn_id": nid})
            if bool(_get(n, "accepted", default=False)):
                # accepted does not necessarily mean 'winner' but we can tag if role says so
                role = _node_role(n)
                if role == "winner":
                    manifest_items.append({"event": "winner", "bn_id": nid})
            tags = _get(n, "tags") or []
            if isinstance(tags, str):
                try:
                    import json as _json
                    tags = _json.loads(tags)
                except Exception:
                    tags = [tags]
            if "rejected" in [str(t).lower() for t in tags]:
                manifest_items.append({"event": "reject", "bn_id": nid})

        for e in edges:
            s = _sid(_get(e, "src", "source", "src_node_id"))
            t = _sid(_get(e, "dst", "target", "dst_node_id"))
            if s and t:
                manifest_items.append({"event": "edge_add", "src": s, "dst": t})

    return build_frames_from_manifest(
        manifest_items=manifest_items,
        nodes=nodes,
        edges=edges,
        positions=positions,
        **kwargs,
    )
