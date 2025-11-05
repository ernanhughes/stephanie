from __future__ import annotations
from typing import Dict, List, Any

def to_cytoscape_elements(nodes: Dict[str, Any], edges: List[Any]) -> dict:
    """
    Convert {id -> NexusNode}, [NexusEdge] -> {'nodes': [...], 'edges': [...]}
    for the /nexus/run/{run_id}/graph.json endpoint.
    """
    cy_nodes = []
    for nid, n in nodes.items():
        label = getattr(n, "title", None) or getattr(n, "text", "") or nid
        cy_nodes.append({
            "data": {
                "id": nid,
                "label": label[:120],
                "type": getattr(n, "target_type", "unknown"),
                "deg": getattr(n, "degree", 0),
            }
        })

    cy_edges = []
    for e in edges:
        cy_edges.append({
            "data": {
                "id": f"{e.src}->{e.dst}",
                "source": e.src,
                "target": e.dst,
                "type": getattr(e, "type", "link"),
                "weight": float(getattr(e, "weight", 0.0) or 0.0),
            }
        })

    return {"nodes": cy_nodes, "edges": cy_edges}
