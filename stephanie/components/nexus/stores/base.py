# stephanie/components/nexus/store/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..app.types import NexusEdge, NexusNode, NodeId  # was .types


class GraphStore(ABC):
    @abstractmethod
    def upsert_node(self, node: NexusNode) -> None: ...
    @abstractmethod
    def add_edges(self, edges: List[NexusEdge]) -> None: ...
    @abstractmethod
    def neighbors(self, node_id: NodeId) -> List[NexusEdge]: ...
    @abstractmethod
    def get(self, node_id: NodeId) -> Optional[NexusNode]: ...
    @abstractmethod
    def has(self, node_id: NodeId) -> bool: ...
