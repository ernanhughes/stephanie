# stephanie/components/nexus/viewer/exporters.py
from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Dict, List

from pyvis.network import Network

from stephanie.components.nexus.types import NexusEdge, NexusNode
from stephanie.utils.json_sanitize import dumps_safe


def export_pyvis_html_rich(output_path: str, *, nodes, edges, positions, title: str = ""):
    """
    Deterministic export using vis-network with:
      - fixed positions (physics off) using provided `positions`
      - node size ~= degree
      - edge color by type (MST vs KNN)
      - dark theme + legend + edge toggles
      - JS-side scaling of normalized coords + fit on load/resize
    """
    # degree for node sizing
    deg = {str(nid): 0 for nid in nodes.keys()}

    def _nid(x):
        return str(getattr(x, "source", getattr(x, "src", "")))

    def _tid(x):
        return str(getattr(x, "target", getattr(x, "dst", "")))

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
                "label": getattr(n, "title", None)
                or getattr(n, "text", "")[:80]
                or str(nid),
                "shape": "dot",
                "color": "#90caf9",
                "value": max(3, min(25, deg.get(str(nid), 1) + 3)),
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
                "physics": False,
                "font": {"color": "#eee"},
            }
        )

    def _edge_type(e) -> str:
        et = getattr(e, "type", None)
        if et:
            return str(et)
        t = getattr(e, "title", "") or getattr(e, "label", "")
        tl = t.lower()
        if "mst" in tl:
            return "backbone_mst"
        if "knn" in tl:
            return "knn_blend"
        return "edge"

    def _edge_weight(e) -> float:
        try:
            return float(getattr(e, "weight", 0.0) or 0.0)
        except Exception:
            return 0.0

    j_edges = []
    for e in edges:
        et = _edge_type(e)
        color = "#f07178" if et == "backbone_mst" else "#56b6c2"  # mst red-ish, knn teal
        w = _edge_weight(e)
        j_edges.append(
            {
                "from": _nid(e),
                "to": _tid(e),
                "arrows": "to",
                "color": color,
                "width": 3 if w >= 0.9 else 2 if w >= 0.8 else 1,
                "title": f"{et} ({w:.3f})" if w else et,
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
        <span><span class="chip" style="background:#f07178"></span>MST edges</span>
        <span><span class="chip" style="background:#56b6c2"></span>KNN edges</span>
      </div>
      <div style="margin-left:auto; display:flex; gap:12px;">
        <label><input type="checkbox" id="toggleKnn" checked> Show KNN</label>
        <label><input type="checkbox" id="toggleMst" checked> Show MST</label>
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
  const edgesKNN = edgesRaw.filter(e => (e.title||"").toLowerCase().includes("knn"));
  const edgesMST = edgesRaw.filter(e => (e.title||"").toLowerCase().includes("mst"));
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
</script>
</body>
</html>"""
    Path(output_path).write_text(html, encoding="utf-8")


def export_pyvis_html(
    nodes: Dict[str, NexusNode],
    edges: List[NexusEdge],
    output_path: str,
    title: str,
) -> str:
    from pathlib import Path
    import json
    from pyvis.network import Network

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

    # Thinner edges + smaller arrows; allow per-edge rgba colors
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
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.45}},  # smaller arrowheads
            "smooth": {"type": "dynamic"},
            "color": {"inherit": False},   # we’ll send per-edge rgba
            "selectionWidth": 1,
            "hoverWidth": 1
        }
    }
    net.set_options(json.dumps(options))

    # Nodes
    for nid, n in nodes.items():
        nid_s = str(nid)
        label_src = getattr(n, "title", None) or getattr(n, "text", "") or nid_s
        label = str(label_src)[:80]
        degree = int(getattr(n, "degree", 1) or 1)
        size = max(6, min(18, int((degree ** 0.5) * 7)))
        net.add_node(
            nid_s,
            label=label,
            title=str(getattr(n, "target_type", "node")),
            value=size,
        )

    # Edge styling helpers
    def _edge_rgba_and_width(etype: str, w: float):
        et = (etype or "").lower()
        # softer red for temporal/MST, desaturated gray-blue for KNN, neutral gray for other
        if "temporal" in et or "mst" in et:
            col = "rgba(240,113,120,0.85)"   # soft red (was bright red)
        elif "knn" in et:
            col = "rgba(138,162,178,0.60)"   # gray-blue (less neon)
        else:
            col = "rgba(107,114,128,0.50)"   # neutral gray
        # much thinner lines overall
        width = max(0.6, min(1.4, 0.8 + (w or 0.0) * 0.8))
        return col, width

    # Edges
    for e in edges:
        etype = getattr(e, "type", "edge")
        w = float(getattr(e, "weight", 0.0) or 0.0)
        src, dst = str(getattr(e, "src", "")), str(getattr(e, "dst", ""))
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


def export_graph_json(
    path,
    nodes: Dict[str, NexusNode],
    edges: List[NexusEdge],
    positions: dict | None = None,
):
    """
    Export a Cytoscape-friendly graph.json with optional positions.
    """
    elements = {"nodes": [], "edges": []}

    for nid, n in nodes.items():
        d = {
            "id": nid,
            "label": getattr(n, "title", None)
            or getattr(n, "text", "")[:80]
            or nid,
            "type": getattr(n, "target_type", "unknown"),
            "deg": int(getattr(n, "degree", 0) or 0),
        }
        if positions and nid in positions:
            x, y = positions[nid]
            d["x"], d["y"] = x, y
        elements["nodes"].append({"data": d})

    for e in edges:
        elements["edges"].append(
            {
                "data": {
                    "id": f"{e.src}->{e.dst}",
                    "source": e.src,
                    "target": e.dst,
                    "type": e.type,
                    "weight": float(getattr(e, "weight", 0.0) or 0.0),
                }
            }
        )

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(dumps_safe(elements), encoding="utf-8")
