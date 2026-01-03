# stephanie/services/knowledge_graph/explorer/explorer_graph.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterable
from enum import Enum


class NodeType(str, Enum):
    PLAN_TRACE = "plan_trace"
    EXEC_STEP = "execution_step"
    EVALUATION = "evaluation"
    SCORE = "score"
    SCORABLE = "scorable"          # future generic
    CASE = "case"                  # future
    CANDIDATE = "candidate"        # future
    EVIDENCE_CARD = "evidence_card" # future
    BUNDLE = "bundle"              # future
    DRAFT = "draft_variant"        # future
    ARENA_MATCH = "arena_match"    # future


class EdgeType(str, Enum):
    HAS_STEP = "has_step"
    NEXT = "next"
    HAS_EVALUATION = "has_evaluation"
    HAS_SCORE = "has_score"


def make_node_id(node_type: str, node_pk: Any) -> str:
    # Stable canonical identity in the graph
    return f"{node_type}:{node_pk}"


@dataclass
class ExplorerNode:
    node_id: str
    node_type: str
    label: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExplorerNode":
        return ExplorerNode(
            node_id=d["node_id"],
            node_type=d["node_type"],
            label=d.get("label", ""),
            meta=d.get("meta", {}) or {},
        )


@dataclass
class ExplorerEdge:
    src: str
    dst: str
    edge_type: str
    weight: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "src": self.src,
            "dst": self.dst,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExplorerEdge":
        return ExplorerEdge(
            src=d["src"],
            dst=d["dst"],
            edge_type=d["edge_type"],
            weight=d.get("weight"),
            meta=d.get("meta", {}) or {},
        )


@dataclass
class ExplorerGraph:
    root_id: str
    nodes: Dict[str, ExplorerNode] = field(default_factory=dict)
    edges: List[ExplorerEdge] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: ExplorerNode) -> ExplorerNode:
        self.nodes[node.node_id] = node
        return node

    def add_edge(self, edge: ExplorerEdge) -> ExplorerEdge:
        self.edges.append(edge)
        return edge

    def get_or_add_node(self, node_id: str, node_type: str, label: str = "", meta: Optional[dict] = None) -> ExplorerNode:
        if node_id in self.nodes:
            return self.nodes[node_id]
        return self.add_node(ExplorerNode(node_id=node_id, node_type=node_type, label=label, meta=meta or {}))

    def to_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExplorerGraph":
        g = ExplorerGraph(root_id=d["root_id"], meta=d.get("meta", {}) or {})
        for nd in d.get("nodes", []):
            g.add_node(ExplorerNode.from_dict(nd))
        for ed in d.get("edges", []):
            g.add_edge(ExplorerEdge.from_dict(ed))
        return g
