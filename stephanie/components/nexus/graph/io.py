from __future__ import annotations

from typing import Any, Dict, List

# Expect NexusNode/NexusEdge to be simple dataclasses or objects with attributes used below


def to_cytoscape_json(nodes: Dict[str, Any], edges: List[Any]) -> dict:
    """
    Build Cytoscape.js-compatible JSON.

    Input:
      - nodes: {id -> object} where object may be a dataclass or any object/dict
               with attributes/keys: title | text | target_type | weight | x/y/position
      - edges: [edge,...] where edge may be object/dict with: src|source, dst|target, type, weight

    Output (deterministic id-sorted):
      {
        "elements": {
          "nodes": [{"data": {...}, "position": {...}?}, ...],
          "edges": [{"data": {...}}, ...]
        }
      }
    """

    def _get(obj, *names, default=None):
        # Try attributes, then dict keys
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return v
            if isinstance(obj, dict) and n in obj and obj[n] is not None:
                return obj[n]
        return default

    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return float(default)

    # ---- degree map (by node id) ----
    deg = {str(nid): 0 for nid in nodes.keys()}

    def _eid_src(e):
        return _get(e, "src", "source", "from", default=None)

    def _eid_dst(e):
        return _get(e, "dst", "target", "to", default=None)

    for e in edges:
        s = str(_eid_src(e)) if _eid_src(e) is not None else None
        t = str(_eid_dst(e)) if _eid_dst(e) is not None else None
        if s in deg:
            deg[s] += 1
        if t in deg:
            deg[t] += 1

    # ---- nodes ----
    cy_nodes: List[dict] = []
    for nid, n in sorted(nodes.items(), key=lambda kv: str(kv[0])):
        sid = str(nid)

        title = _get(n, "title", default=None)
        text = _get(n, "text", default=None)
        label = str(title or (str(text)[:80] if text else sid))

        ntype = _get(n, "target_type", default="node")
        weight = _to_float(_get(n, "weight", default=0.0))

        # Optional fixed position, if present on node
        # Accept: (x,y) tuple in 'pos', or numeric attrs x/y, or dict 'position'
        pos = _get(n, "position", default=None)
        if pos is None:
            pos_tuple = _get(n, "pos", default=None)
            if isinstance(pos_tuple, (tuple, list)) and len(pos_tuple) == 2:
                pos = {
                    "x": _to_float(pos_tuple[0]),
                    "y": _to_float(pos_tuple[1]),
                }
            else:
                x = _get(n, "x", default=None)
                y = _get(n, "y", default=None)
                if x is not None and y is not None:
                    pos = {"x": _to_float(x), "y": _to_float(y)}

        node_entry = {
            "data": {
                "id": sid,
                "label": label,
                "type": str(ntype),
                "deg": int(deg.get(sid, 0)),
                "weight": weight,
            }
        }
        if isinstance(pos, dict) and "x" in pos and "y" in pos:
            node_entry["position"] = {
                "x": _to_float(pos["x"]),
                "y": _to_float(pos["y"]),
            }

        cy_nodes.append(node_entry)

    # ---- edges ----
    cy_edges: List[dict] = []
    for e in edges:
        s = _eid_src(e)
        t = _eid_dst(e)
        if s is None or t is None:
            continue
        s, t = str(s), str(t)

        etype = _get(e, "type", default="edge")
        weight = _to_float(_get(e, "weight", default=0.0))

        cy_edges.append(
            {
                "data": {
                    "id": f"{s}->{t}",
                    "source": s,
                    "target": t,
                    "type": str(etype),
                    "weight": weight,
                }
            }
        )

    return {"elements": {"nodes": cy_nodes, "edges": cy_edges}}
