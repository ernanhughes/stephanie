# stephanie/components/nexus/store/dict_store.py
from __future__ import annotations

from __future__ import annotations
from typing import Dict, List
from ..types import NexusNode, NexusEdge, NodeId

class NexusGraphStore:
    """In-memory store (swap to SQLAlchemy later)."""
    def __init__(self) -> None:
        self.nodes: Dict[NodeId, NexusNode] = {}
        self.edges_by_src: Dict[NodeId, List[NexusEdge]] = {}

    def upsert_node(self, n: NexusNode) -> None:
        self.nodes[n.node_id] = n

    def add_edges(self, edges: List[NexusEdge]) -> None:
        for e in edges:
            self.edges_by_src.setdefault(e.src, []).append(e)

    def neighbors(self, node_id: NodeId) -> List[NexusEdge]:
        return self.edges_by_src.get(node_id, [])

    def get(self, node_id: NodeId) -> NexusNode:
        return self.nodes[node_id]

    def has(self, node_id: NodeId) -> bool:
        return node_id in self.nodes
