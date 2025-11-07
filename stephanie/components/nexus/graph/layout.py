# stephanie/components/nexus/graph/layout.py
from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

from stephanie.components.nexus.app.types import NexusEdge, NexusNode


def compute_positions(nodes: Dict[str, NexusNode], edges: List[NexusEdge]) -> Dict[str, Tuple[float,float]]:
    G = nx.Graph()
    G.add_nodes_from(nodes.keys())
    G.add_edges_from([(e.src, e.dst) for e in edges])

    # Spring layout with deterministic seed; k tuned for speed
    pos = nx.spring_layout(G, k=None, iterations=100, seed=42, dim=2)  # k auto-scales by nx
    return { nid: (float(x), float(y)) for nid, (x,y) in pos.items() }
