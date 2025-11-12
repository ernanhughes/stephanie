# nexus/graph/to_nx.py
from typing import Dict, List, Any, Tuple
import networkx as nx

def to_networkx(nodes: Dict[str, Any], edges: List[Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    for node_id, n in nodes.items():
        # Accept either dataclass or mapping
        data = n.__dict__ if hasattr(n, "__dict__") else dict(n)
        G.add_node(node_id, **data)
    for e in edges:
        data = e.__dict__ if hasattr(e, "__dict__") else dict(e)
        u = data.get("source") or data.get("src") or data.get("u")
        v = data.get("target") or data.get("dst") or data.get("v")
        if u is None or v is None:
            continue
        G.add_edge(u, v, **data)
    return G
