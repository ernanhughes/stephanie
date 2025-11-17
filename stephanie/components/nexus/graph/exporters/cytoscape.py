# stephanie/components/nexus/graph/exporters/cytoscape.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from .base import BaseGraphExporter

if TYPE_CHECKING:
    from stephanie.components.nexus.graph.graph import NexusGraph, NexusNode
    from stephanie.models.nexus import NexusEdgeORM


# --------------------------------------------------------------------------- #
# Low-level helpers: generic nodes/edges → Cytoscape elements                 #
# --------------------------------------------------------------------------- #

def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """
    Try attributes, then dict keys, in order. Stop at first non-None.
    """
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
        if isinstance(obj, dict) and n in obj and obj[n] is not None:
            return obj[n]
    return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _eid_src(e: Any) -> Any:
    return _get(e, "src", "source", "from", default=None)


def _eid_dst(e: Any) -> Any:
    return _get(e, "dst", "target", "to", default=None)


def to_cytoscape_json(nodes: Dict[str, Any], edges: List[Any]) -> Dict[str, Any]:
    """
    Build Cytoscape.js-compatible JSON.

    Input:
      - nodes: {id -> object} where object may be a dataclass or any object/dict
               with attributes/keys: title | text | target_type | weight | x/y/position
      - edges: [edge,...] where edge may be object/dict with: src|source, dst|target,
               type, weight

    Output (deterministic id-sorted):
      {
        "elements": {
          "nodes": [{"data": {...}, "position": {...}?}, ...],
          "edges": [{"data": {...}}, ...]
        }
      }
    """

    # ---- degree map (by node id) ----
    deg = {str(nid): 0 for nid in nodes.keys()}

    for e in edges:
        s = _eid_src(e)
        t = _eid_dst(e)
        s = str(s) if s is not None else None
        t = str(t) if t is not None else None
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

        ntype = _get(n, "target_type", "type", default="node")
        weight = _to_float(_get(n, "weight", default=0.0))

        # Optional fixed position, if present on node
        # Accept: (x, y) tuple in 'pos', or numeric attrs x/y, or dict 'position'
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


def to_cytoscape_elements(nodes: Dict[str, Any], edges: List[Any]) -> Dict[str, Any]:
    """
    Legacy helper: return {'nodes': [...], 'edges': [...]} only.

    Kept for compatibility with any code that expects just the elements dict.
    """
    packed = to_cytoscape_json(nodes, edges)
    return packed["elements"]

# still in cytoscape.py

def _graph_to_nodes_edges(graph: "NexusGraph") -> tuple[Dict[str, Any], List[Any]]:
    """
    Pull NexusNodes and NexusEdgeORM objects out of a NexusGraph
    and normalise into {id -> node}, [edge,...].
    """
    nodes: Dict[str, Any] = {}
    for node in graph.iter_nodes_for_run():
        # node is a NexusNode dataclass, keyed by its id
        nodes[node.id] = node

    edges = graph.list_edges()  # [NexusEdgeORM,...]
    return nodes, edges


class CytoscapeGraphExporter(BaseGraphExporter):
    """
    Export a NexusGraph as Cytoscape.js JSON and a minimal HTML viewer.

    This is intentionally dumb/simple: layout is left to Cytoscape's built-in
    algorithms (default: 'cose') and styling is minimal.
    """

    def __init__(self, *, name: str = "cytoscape"):
        super().__init__(name=name)

    # ---- JSON payload ------------------------------------------------------

    def build_payload(self, graph: "NexusGraph") -> Dict[str, Any]:
        nodes, edges = _graph_to_nodes_edges(graph)
        elements = to_cytoscape_elements(nodes, edges)
        return {
            "run_id": graph.run_id,
            "elements": elements,
        }

    # ---- HTML writer -------------------------------------------------------

    def write_html(
        self,
        graph: "NexusGraph",
        out_path: Path,
        *,
        title: str = "Nexus graph",
    ) -> Path:
        """
        Write a self-contained HTML page that embeds Cytoscape.js
        and renders the graph for this run.
        """
        payload = self.build_payload(graph)
        elements = payload["elements"]

        import json

        elements_json = json.dumps(elements, ensure_ascii=False)

        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    html, body, #cy {{
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
    }}
  </style>
  <script src="https://unpkg.com/cytoscape@3.29.2/dist/cytoscape.min.js"></script>
</head>
<body>
  <div id="cy"></div>
  <script>
    const elements = {elements_json};

    const cy = cytoscape({{
      container: document.getElementById('cy'),
      elements: elements,
      layout: {{ name: 'cose' }},
      style: [
        {{
          selector: 'node',
          style: {{
            'label': 'data(label)',
            'font-size': 8,
            'background-opacity': 1,
            'background-color': '#999',
            'border-width': 0.5,
            'border-color': '#444'
          }}
        }},
        {{
          selector: 'edge',
          style: {{
            'width': 0.5,
            'line-color': '#bbb',
            'curve-style': 'bezier'
          }}
        }}
      ]
    }});
  </script>
</body>
</html>
"""

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        return out_path
