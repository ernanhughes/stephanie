# stephanie/components/nexus/viewer/exporters.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
from pyvis.network import Network

from stephanie.components.nexus.types import NexusNode, NexusEdge
from stephanie.utils.json_sanitize import dumps_safe  # used by export_graph_json

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
        bgcolor="#111",
        font_color="#eee",
    )
    net.toggle_physics(True)

    # IMPORTANT: pass valid JSON (keys quoted, numbers are numbers)
    options = {
        "physics": {
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 250}
        },
        "nodes": {
            "shape": "dot",
            "scaling": {"min": 3, "max": 25}
        },
        "edges": {
            "smooth": {"type": "dynamic"}
        }
    }
    net.set_options(json.dumps(options))

    # Nodes
    for nid, n in nodes.items():
        nid_s = str(nid)
        label_src = getattr(n, "title", None) or getattr(n, "text", "") or nid_s
        label = str(label_src)[:80]
        degree = int(getattr(n, "degree", 1) or 1)
        size = max(6, min(22, int((degree ** 0.5) * 8)))
        net.add_node(
            nid_s,
            label=label,
            title=str(getattr(n, "target_type", "node")),
            value=size,
        )

    # Edges
    for e in edges:
        etype = getattr(e, "type", "edge")
        w = float(getattr(e, "weight", 0.0) or 0.0)
        src, dst = str(getattr(e, "src", "")), str(getattr(e, "dst", ""))
        color = "#5ec269" if etype == "temporal_next" else "#56b6c2"
        width = 2 if etype == "temporal_next" else max(1, int(round(w * 3)))
        net.add_edge(src, dst, title=f"{etype} ({w:.3f})", color=color, width=width)

    # Write file (no auto-open)
    net.write_html(str(out), notebook=False)
    return str(out)
