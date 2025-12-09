from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Reference + paper graph structs
# ---------------------------------------------------------------------------


@dataclass
class ReferenceRecord:
    """
    Lightweight representation of a reference to an arXiv paper.

    This is intentionally generic: you can populate it from arXiv API,
    OpenAlex, or your own reference extractor.
    """

    arxiv_id: str
    title: Optional[str] = None
    url: Optional[str] = None
    source: str = "unknown"  # e.g. "arxiv", "openalex", "parsed_pdf"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperNode:
    """
    Node in the paper reference graph.

    role:
        - "root"              -> the paper you asked to process
        - "reference"         -> cited by root
        - "similar_root"      -> HF similar to root
        - "similar_reference" -> HF similar to a reference
    """

    arxiv_id: str
    role: str
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    published_date: Optional[str] = None
    abstract: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[str] = None

    pdf_path: Optional[Path] = None  # local copy of the PDF
    text_hash: Optional[str] = None  # hash of extracted text (if you want)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceEdge:
    """
    Directed edge in the paper reference graph.

    kind:
        - "cites"       -> src cites dst
        - "similar_to"  -> src is similar to dst
    """

    src: str  # arxiv_id
    dst: str  # arxiv_id
    kind: str
    weight: float = 1.0


@dataclass
class PaperReferenceGraph:
    """
    In-memory graph of root paper + references + similar papers.
    """

    root_arxiv_id: str
    nodes: Dict[str, PaperNode] = field(default_factory=dict)  # arxiv_id -> node
    edges: List[ReferenceEdge] = field(default_factory=list)

    # Convenience
    def add_node(self, node: PaperNode) -> None:
        if not node.arxiv_id:
            raise ValueError("PaperNode requires arxiv_id")
        self.nodes[node.arxiv_id] = node

    def get_node(self, arxiv_id: str) -> Optional[PaperNode]:
        return self.nodes.get(arxiv_id)

    def add_edge(self, src: str, dst: str, kind: str, weight: float = 1.0) -> None:
        if src == dst:
            return
        self.edges.append(ReferenceEdge(src=src, dst=dst, kind=kind, weight=weight))

    def neighbors(self, arxiv_id: str, kind: Optional[str] = None) -> List[PaperNode]:
        out: List[PaperNode] = []
        for e in self.edges:
            if e.src == arxiv_id and (kind is None or e.kind == kind):
                node = self.nodes.get(e.dst)
                if node:
                    out.append(node)
        return out


# ---------------------------------------------------------------------------
# Section + matching structs
# ---------------------------------------------------------------------------


@dataclass
class DocumentSection:
    """
    A section / chunk of a paper's text.

    The idea is:
    - Everything is broken into sections of reasonable size.
    - You can attach summary + title + embedding later.
    """

    id: str  # e.g. f"{paper_arxiv_id}::sec-{section_index}"
    paper_arxiv_id: str
    paper_role: str  # same semantics as PaperNode.role
    section_index: int

    text: str

    title: Optional[str] = None        # summarizer-generated
    summary: Optional[str] = None      # summarizer-generated

    start_char: Optional[int] = None   # optional, for traceability
    end_char: Optional[int] = None

    # You can decide whether to store raw vectors, embedding IDs, etc.
    embedding: Any = None

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionMatch:
    """
    Link between two sections (usually: root section -> related section).
    """

    source_section_id: str
    target_section_id: str
    score: float
    rank: int
    reason: Optional[str] = None  # free-form explanation, if you want


@dataclass
class ConceptCluster:
    """
    A cluster of sections that share a concept (roughly).

    This can be produced by simple greedy clustering or something more
    sophisticated; for now it's just a convenient container.
    """

    cluster_id: str
    section_ids: List[str]
    score: float = 1.0
    label: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
