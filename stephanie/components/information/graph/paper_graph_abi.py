from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphNode:
    id: str
    type: str  # "paper" | "section" | "ref" | "element" | ...
    title: Optional[str] = None
    paper_id: Optional[str] = None
    section_id: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    src: str
    dst: str
    type: str  # "SIMILAR_SECTION" | "CITES" | "ENTITY_OVERLAP"
    weight: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperGraphABI:
    version: str
    run_id: str
    root_arxiv_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
