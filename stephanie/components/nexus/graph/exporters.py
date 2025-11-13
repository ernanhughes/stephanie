# stephanie/components/nexus/viewer/exporters.py
from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pyvis.network import Network

from stephanie.components.nexus.app.types import NexusEdge, NexusNode
from stephanie.utils.json_sanitize import dumps_safe

# =============================================================================
# Safe helpers
# =============================================================================

def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely try multiple attribute or dict keys in order; never raises.
    Example: _get(node, "quality", "score", default=None)
    """
    if obj is None:
        return default
    for k in keys:
        try:
            if isinstance(obj, dict):
                v = obj.get(k, None)
            else:
                v = getattr(obj, k, None)  # attr-safe (no AttributeError)
        except Exception:
            v = None
        if v is not None:
            return v
    return default


def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default


def _metrics_lookup(node: Any, key: str) -> Optional[float]:
    """
    Try to read node.metrics.vector[key] whether metrics is a dict or an object.
    """
    m = getattr(node, "metrics", None)
    if m is None:
        return None
    vec = None
    if isinstance(m, dict):
        vec = m.get("vector") if isinstance(m.get("vector"), dict) else None
    else:
        vec = getattr(m, "vector", None)
    if isinstance(vec, dict):
        return vec.get(key)
    return None


def _as_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: Optional[int] = 0) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _edge_type(e: Any) -> str:
    et = _get(e, "type", "etype", "label", "title", default="edge")
    if not isinstance(et, str):
        try:
            et = str(et)
        except Exception:
            et = "edge"
    return et


def _edge_weight(e: Any) -> float:
    return _as_float(_get(e, "weight", "w", default=0.0), default=0.0) or 0.0


def _node_degree(n: Any) -> int:
    return _as_int(_get(n, "degree", "deg", default=0), default=0) or 0


def _node_label(nid: str, n: Any) -> str:
    label_src = _coalesce(
        _get(n, "title", default=None),
        _get(n, "text", default=None),
        nid,
    )
    try:
        label = str(label_src)
    except Exception:
        label = nid
    return label[:80]


# =============================================================================
# Visual helpers (PyVis / HTML)
# =============================================================================

def _edge_rgba_and_width(etype: str, w: float) -> Tuple[str, float]:
    et = (etype or "").lower()
    # Red-ish for temporal/MST, teal-ish for KNN, soft gray default
    if "temporal" in et or "mst" in et or "backbone" in et:
        col = "rgba(240,113,120,0.85)"   # soft red
    elif "knn" in et:
        col = "rgba(138,162,178,0.60)"   # gray-blue
    else:
        col = "rgba(107,114,128,0.50)"   # neutral gray
    width = max(0.6, min(1.6, 0.8 + (w or 0.0) * 0.8))
    return col, width


# =============================================================================
# Rich (deterministic layout) HTML exporter
# =============================================================================

def export_pyvis_html_rich(
    output_path: str,
    *,
    nodes: Dict[str, NexusNode],
    edges: List[NexusEdge],
    positions: Dict[str, Tuple[float, float]],
    title: str = "",
) -> str:
    """
    Deterministic export using vis-network with:
      - fixed positions (physics off) using provided `positions`
      - node size ~= degree
      - edge color by type (MST vs KNN vs other)
      - dark theme + legend + edge toggles
      - JS-side scaling of normalized coords + fit on load/resize
    """
    # degree for node sizing
    deg = {str(nid): 0 for nid in nodes.keys()}

    def _nid(x):
        return str(_get(x, "source", "src", default=""))

    def _tid(x):
        return str(_get(x, "target", "dst", default=""))

    for e in edges:
        s, t = _nid(e), _tid(e)
        if s in deg:
            deg[s] += 1
        if t in deg:
            deg[t] += 1

    # nodes → JSON-serializable objects (x/y may be small/normalized)
    j_nodes = []
    for nid, n in nodes.items():
        x, y = positions.get(nid, (None, None))
        j_nodes.append(
            {
                "id": str(nid),
                "label": _node_label(str(nid), n),
                "shape": "dot",
                "color": "#90caf9",
                "value": max(3, min(25, deg.get(str(nid), 1) + 3)),
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
                "physics": False,
                "font": {"color": "#eee"},
            }
        )

    j_edges = []
    for e in edges:
        et = _edge_type(e)
        color, width = _edge_rgba_and_width(et, _edge_weight(e))
        j_edges.append(
            {
                "from": _get(e, "source", "src", default=""),
                "to": _get(e, "target", "dst", default=""),
                "arrows": "to",
                "color": color,
                "width": width if width else 1,
                "title": f"{et} ({_edge_weight(e):.3f})",
                "etype": et,
            }
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/10.0.2/dist/dist/vis-network.min.css"/>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/10.0.2/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ background:#0e0e0e; color:#ddd; margin:0; font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial; }}
    #wrap {{ display:flex; flex-direction:column; height:100vh; }}
    header {{ padding:10px 14px; border-bottom:1px solid #333; }}
    #mynetwork {{ flex:1 1 auto; background:#111; }}
    .legend {{ display:flex; gap:16px; align-items:center; font-size:14px; }}
    .chip {{ display:inline-block; width:14px; height:14px; border-radius:3px; margin-right:6px; }}
    .panel {{ display:flex; gap:18px; align-items:center; }}
    .muted {{ color:#aaa; }}
    label {{ user-select:none; }}
  </style>
</head>
<body>
<div id="wrap">
  <header>
    <div class="panel">
      <strong>{title}</strong>
      <div class="legend">
        <span><span class="chip" style="background:#90caf9"></span>node size = degree</span>
        <span><span class="chip" style="background:#f07178"></span>MST/temporal</span>
        <span><span class="chip" style="background:#8AA2B2"></span>KNN</span>
      </div>
      <div style="margin-left:auto; display:flex; gap:12px;">
        <label><input type="checkbox" id="toggleKnn" checked> Show KNN</label>
        <label><input type="checkbox" id="toggleMst" checked> Show MST/Temporal</label>
      </div>
    </div>
  </header>
  <div id="mynetwork"></div>
</div>

<script>
  // Scale normalized coordinates (e.g., -1..1) up to pixels
  const SCALE = 600; // tweak 400–900 depending on density

  const rawNodes = {json.dumps(j_nodes)};
  // apply scaling & freeze positions
  rawNodes.forEach(n => {{
    if (typeof n.x === 'number') n.x *= SCALE;
    if (typeof n.y === 'number') n.y *= SCALE;
    n.fixed = {{ x: true, y: true }};
    n.physics = false;
  }});
  const nodes = new vis.DataSet(rawNodes);

  const edgesRaw = {json.dumps(j_edges)};
  const edgesKNN = edgesRaw.filter(e => (e.etype||"").toLowerCase().includes("knn"));
  const edgesMST = edgesRaw.filter(e => {{
    const t = (e.etype||"").toLowerCase();
    return t.includes("mst") || t.includes("temporal") || t.includes("backbone");
  }});
  const edges = new vis.DataSet(edgesRaw);

  const container = document.getElementById('mynetwork');
  const data = {{ nodes, edges }};
  const options = {{
    physics: false,
    interaction: {{ hover: false, zoomView: true, dragView: true }},
    nodes: {{ shape:'dot', scaling: {{ min:3, max:25 }} }},
    edges: {{ smooth: false }}, // perf-friendly on Windows
    layout: {{ improvedLayout: false }} // skip extra layout passes
  }};

  const network = new vis.Network(container, data, options);

  // Frame content on first draw + when resizing
  network.once('afterDrawing', () => network.fit({{ animation: false }}));
  window.addEventListener('resize', () => network.fit({{ animation: false }}));

  // Edge toggles
  const toggleKnn = document.getElementById('toggleKnn');
  const toggleMst = document.getElementById('toggleMst');
  function refreshEdges() {{
    const next = [];
    if (toggleKnn.checked) next.push(...edgesKNN);
    if (toggleMst.checked) next.push(...edgesMST);
    edges.clear();
    edges.add(next);
  }}
  toggleKnn.addEventListener('change', refreshEdges);
  toggleMst.addEventListener('change', refreshEdges);
  refreshEdges();
</script>
</body>
</html>"""
    Path(output_path).write_text(html, encoding="utf-8")
    return str(output_path)


# =============================================================================
# Simple PyVis exporter (force layout)
# =============================================================================

def export_pyvis_html(
    nodes: Dict[str, NexusNode],
    edges: List[NexusEdge],
    output_path: str,
    title: str,
) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        height="100vh",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#0e0e0e",          # black background
        font_color="#e5e7eb",       # light gray labels
    )

    options = {
        "physics": {
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 250}
        },
        "interaction": {"hover": True, "zoomView": True, "dragView": True},
        "nodes": {
            "shape": "dot",
            "scaling": {"min": 3, "max": 20},
            "font": {"color": "#e5e7eb"}
        },
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.45}},
            "smooth": {"type": "dynamic"},
            "color": {"inherit": False},
            "selectionWidth": 1,
            "hoverWidth": 1
        }
    }
    net.set_options(json.dumps(options))

    # Nodes
    for nid, n in nodes.items():
        nid_s = str(nid)
        label = _node_label(nid_s, n)
        degree = _node_degree(n)
        size = max(6, min(18, int((degree ** 0.5) * 7)))
        net.add_node(
            nid_s,
            label=label,
            title=str(_get(n, "target_type", default="node")),
            value=size,
        )

    # Edges
    for e in edges:
        etype = _edge_type(e)
        w = _edge_weight(e)
        src, dst = str(_get(e, "src", "source", default="")), str(_get(e, "dst", "target", default=""))
        color, width = _edge_rgba_and_width(etype, w)
        net.add_edge(
            src,
            dst,
            title=f"{etype} ({w:.3f})",
            color=color,
            width=width,
            arrows="to",
        )

    net.write_html(str(out), notebook=False)
    return str(out)


# =============================================================================
# Cytoscape-friendly JSON exporter
# =============================================================================

def export_graph_json(
    path: Union[str, Path],
    nodes: Dict[str, NexusNode],
    edges: List[NexusEdge],
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    *,
    include_channels: bool = False,
    include_node_text: bool = True,
    include_metrics: bool = True,
    extra_node_fields: Optional[List[str]] = None,
    extra_edge_fields: Optional[List[str]] = None,
) -> str:
    """
    Export a Cytoscape-friendly graph.json with optional positions and channels.

    elements = {
      "nodes": [{"data": {...}, "position": {"x":..,"y":..}} ...],
      "edges": [{"data": {...}} ...]
    }

    - `include_channels`: if True, copy edge.channels (dict) into edge["data"]["channels"]
    - graceful handling of missing attributes (no exceptions on absent fields)
    - returns the string path written for convenience
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extra_node_fields = list(extra_node_fields or [])
    extra_edge_fields = list(extra_edge_fields or [])

    elements: Dict[str, list] = {"nodes": [], "edges": []}

    # ---------------- Nodes ----------------
    for nid, n in (nodes or {}).items():
        nid_s = str(nid)

        # Quality / score coalesce (safe)
        q = _as_float(
            _coalesce(
                _get(n, "quality", default=None),
                _get(n, "score", default=None),
                _metrics_lookup(n, "quality"),
                _metrics_lookup(n, "hrm"),
                _metrics_lookup(n, "alignment"),
            ),
            default=None,
        )

        d: Dict[str, Any] = {
            "id": nid_s,
            "label": _node_label(nid_s, n),
            "type": _get(n, "target_type", default="unknown"),
            "deg": _node_degree(n),
        }
        if q is not None:
            d["quality"] = float(q)

        if include_node_text:
            txt_preview = _get(n, "text", default=None)
            if txt_preview:
                try:
                    d["text"] = str(txt_preview)[:512]
                except Exception:
                    pass

        if include_metrics:
            # If a vector is present, we can attach a small selection for convenience
            vec = _metrics_lookup(n, "vector")  # returns None by design; we’ll pull manually
            m = getattr(n, "metrics", None)
            vdict = None
            if isinstance(m, dict):
                vdict = m.get("vector") if isinstance(m.get("vector"), dict) else None
            else:
                vdict = getattr(m, "vector", None)

            if isinstance(vdict, dict):
                # Attach small, common keys (don’t bloat JSON)
                subset_keys = ("alignment", "faithfulness", "coverage", "clarity", "coherence", "hrm")
                d["metrics"] = {k: _as_float(vdict.get(k), None) for k in subset_keys if k in vdict}

        # Optional extra node fields
        for k in extra_node_fields:
            v = _get(n, k, default=None)
            if v is not None:
                d[k] = v

        rec: Dict[str, Any] = {"data": d}
        if positions and nid in positions:
            x, y = positions[nid]
            rec["position"] = {"x": float(x), "y": float(y)}
        elements["nodes"].append(rec)

    # ---------------- Edges ----------------
    for e in (edges or []):
        src = _get(e, "src", "source", default=None)
        dst = _get(e, "dst", "target", default=None)
        if src is None or dst is None:
            # Skip malformed edges gracefully
            continue

        etype = _edge_type(e)
        w = _edge_weight(e)

        data: Dict[str, Any] = {
            "id": f"{src}->{dst}",
            "source": str(src),
            "target": str(dst),
            "type": etype,
            "weight": float(w),
        }

        if include_channels:
            ch = _get(e, "channels", default=None)
            if isinstance(ch, dict):
                # Keep as-is, but also add a compact "viz" with only numeric fields for client-side sizing/color
                data["channels"] = ch
                viz = {}
                for k, v in ch.items():
                    fv = _as_float(v, None)
                    if fv is not None:
                        viz[k] = fv
                if viz:
                    data["channels_viz"] = viz

        # Optional extra edge fields
        for k in extra_edge_fields:
            v = _get(e, k, default=None)
            if v is not None:
                data[k] = v

        elements["edges"].append({"data": data})

    out_path.write_text(dumps_safe(elements), encoding="utf-8")
    return str(out_path)
