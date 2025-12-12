from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from stephanie.components.information_legacy.data import DocumentElement, SectionSpine


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
    doi: Optional[str] = None
    year: Optional[str] = None
    url: Optional[str] = None
    raw_citation: Optional[str] = None
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


@dataclass
class SectionNodePayload:
    """
    Minimal payload for inserting a document section into Nexus as a node.
    This is what your Information pipeline will produce.
    """
    id: str                 # stable scorable_id
    text: str
    target_type: str        # e.g. "paper_section"
    paper_id: Optional[str] = None
    section_path: Optional[str] = None  # like "1/1.2/1.2.3"
    domains: List[Any] = field(default_factory=list)
    entities: List[Any] = field(default_factory=list)
    dims: Dict[str, float] = field(default_factory=dict)   # metrics vector
    meta: Dict[str, Any] = field(default_factory=dict)



ElementType = Literal[
    "figure",
    "table",
    "formula_inline",
    "formula_display",
    "chart",
    "text_block",
]


@dataclass
class BoundingBox:
    """
    Simple page-space bounding box.

    Coordinates are in PDF page space (e.g. pixels), with origin (0, 0) at
    the top-left, x to the right, y downwards (or however PaddleOCR-VL reports).
    """
    x1: float
    y1: float
    x2: float
    y2: float

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)




def attach_elements_to_sections(
    sections: Iterable[DocumentSection],
    elements: Iterable[DocumentElement],
) -> List[SectionSpine]:
    """
    Build the 'spine' by attaching extracted elements to their logical sections.

    v1 behavior:
      - For each section, derive (start_page, end_page).
      - Attach every element with element.page in that range.
      - No vertical-position logic yet; that's a v2 refinement.

    Returns:
      List[SectionSpine] in the same order as 'sections'.
    """
    # Materialize inputs (they're often generators)
    section_list = list(sections)
    element_list = list(elements)

    # Pre-index elements by page for cheap lookup
    elements_by_page: Dict[int, List[DocumentElement]] = {}
    for el in element_list:
        elements_by_page.setdefault(el.page, []).append(el)

    spine: List[SectionSpine] = []

    for sec in section_list:
        # --- derive the page range for this section ---
        start_page: Optional[int] = None
        end_page: Optional[int] = None

        # Option 1: explicit attributes
        if hasattr(sec, "start_page") or hasattr(sec, "end_page"):
            sp = getattr(sec, "start_page", None)
            ep = getattr(sec, "end_page", None)
            if sp is not None and ep is not None:
                start_page = int(sp)
                end_page = int(ep)

        # Option 2: list of pages
        if start_page is None and hasattr(sec, "pages"):
            pages = getattr(sec, "pages", None) or []
            try:
                pages = list(pages)
            except TypeError:
                pages = []
            if pages:
                start_page = int(min(pages))
                end_page = int(max(pages))

        # If we still don't know, we can't sensibly attach by page.
        if start_page is None or end_page is None:
            spine.append(SectionSpine(section=sec, elements=[]))
            continue

        # --- collect elements that fall in [start_page, end_page] ---
        attached: List[DocumentElement] = []
        for page in range(start_page, end_page + 1):
            for el in elements_by_page.get(page, []):
                attached.append(el)

        spine.append(
            SectionSpine(
                section=sec,
                elements=attached,
                start_page=start_page,
                end_page=end_page,
            )
        )

    return spine
