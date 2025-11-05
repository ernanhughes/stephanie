# stephanie/components/nexus/viewer/exporters.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
from pyvis.network import Network
from stephanie.components.nexus.types import NexusNode
from stephanie.components.nexus.types import NexusEdge

def export_pyvis_html(nodes: Dict[str, NexusNode], edges: List[NexusEdge], out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    net = Network(height="100vh", width="100%", directed=True, notebook=False, bgcolor="#111", font_color="#eee")
    net.toggle_physics(True)
    net.set_options("""
    const options = {
      physics: { solver: "forceAtlas2Based", stabilization: { iterations: 250 } },
      nodes: { shape: "dot", scaling: { min: 3, max: 25 } },
      edges: { smooth: { type: "dynamic" } }
    };
    """)

    for nid, n in nodes.items():
        label = (n.title or n.text[:80] if getattr(n, "text", None) else nid)
        size  = max(6, min(22, int((getattr(n, "degree", 1) or 1) ** 0.5 * 8)))
        net.add_node(nid, label=label, title=f"{n.target_type}", value=size)

    for e in edges:
        color = "#5ec269" if e.type == "temporal_next" else "#56b6c2"  # green for temporal, teal for knn
        width = 2 if e.type == "temporal_next" else max(1, int((e.weight or 0.1) * 3))
        net.add_edge(e.src, e.dst, title=f"{e.type} ({e.weight:.3f})", color=color, width=width)

    net.show(str(out))  # writes HTML with embedded assets
    return str(out)
