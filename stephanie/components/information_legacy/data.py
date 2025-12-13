# stephanie/models/information.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple

from pyparsing import ABC, abstractmethod


@dataclass
class ReferenceRecord:
    """One cited item in the reference list of a paper."""
    arxiv_id: Optional[str]          # best-case: we know this
    doi: Optional[str]
    title: Optional[str]
    year: Optional[int]
    url: Optional[str]               # PDF url if available
    raw_citation: Optional[str] = None  # fallback text if nothing else

@dataclass
class PaperNode:
    """Node in the citation graph (only need minimal fields for now)."""
    arxiv_id: Optional[str]
    local_dir: Path              # where this paper lives on disk
    pdf_path: Optional[Path]
    metadata: Dict[str, Any]

    # NEW:
    roles: Set[str] = field(default_factory=set)
    # e.g. {"root"}, {"reference"}, {"similar_root"}, {"reference", "similar_ref"}

    importance_score: float = 0.0
    # evidence for scoring (like which seeds it was similar to)
    evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReferenceEdge:
    source_arxiv_id: Optional[str]
    target_arxiv_id: Optional[str]
    relation: str = "cites"

@dataclass
class PaperReferenceGraph:
    root: PaperNode
    nodes: Dict[str, PaperNode]          # keyed by arxiv_id or synthetic id
    edges: List[ReferenceEdge]

@dataclass
class SimilarPaperRecord:
    arxiv_id: Optional[str]
    url: Optional[str]
    title: Optional[str]
    summary: Optional[str] = None
    source: str = "hf_similar"
    score: Optional[float] = None  # if the tool eventually gives one
    raw: Optional[str] = None      # raw text result if needed


class SimilarPaperProvider(ABC):
    @abstractmethod
    def get_similar_for_arxiv(self, arxiv_id: str) -> List[SimilarPaperRecord]:
        raise NotImplementedError
    
@dataclass
class DocumentSection:
    section_id: str
    paper_arxiv_id: Optional[str]
    paper_role: str
    section_index: int
    text: str

    title: Optional[str] = None
    summary: Optional[str] = None

    text_embedding: Optional[list[float]] = None
    title_embedding: Optional[list[float]] = None
    summary_embedding: Optional[list[float]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEW:
    concept_cluster_id: Optional[str] = None
    concept_cluster_strength: float = 0.0  # how “central” this section is to a shared concept


@dataclass
class SectionMatch:
    source_section_id: str     # section in root paper
    target_section_id: str     # section in related paper
    similarity: float
    target_paper_arxiv_id: Optional[str]
    target_paper_role: str

@dataclass
class InformationSource:
    """
    A single source of information for building an 'information object'.

    Examples:
      - kind="document": a local document (paper, blog post, etc.)
      - kind="web":      fetched web page
      - kind="wiki":     Wikipedia article
      - kind="memcube":  existing MemCube content
    """

    kind: str  # "document" | "web" | "wiki" | "memcube" | ...
    id: str  # opaque source id (doc_id, URL, etc.)
    title: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InformationTargetConfig:
    """
    Describes what we are trying to build.

    For the current flow we mainly use:
      - kind="memcube"
      - name/description for the MemCube
      - goal_id / casebook_id for linkage into the rest of Stephanie
      - enable_blog_view=True to signal that we should also produce
        a blog-style markdown view.
    """

    kind: str  # "memcube", later maybe "nexus_page", etc.
    name: str
    description: str

    goal_id: Optional[int] = None
    casebook_id: Optional[int] = None

    enable_blog_view: bool = True

    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InformationRequest:
    """
    Top-level request object passed into InformationProcessor.

    - sources: primary + auxiliary sources we want to synthesize
    - target:  what we want to build (memcube, page, etc.)
    - context: arbitrary pipeline context (goal, document dict, run ids, ...)
    """

    sources: List[InformationSource]
    target: InformationTargetConfig
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InformationResult:
    """
    Result returned by InformationProcessor.

    For now:
      - memcube_id: ID of the created/updated MemCube (if any)
      - bucket_id:  placeholder for future 'bucket' / Nexus graph ids
      - blog_markdown: rendered markdown string (if enable_blog_view=True)
      - topic:   canonical topic/title used for this information object
      - goal_id / casebook_id: copied from target for convenience
      - extra:   free-form debug / telemetry / scoring info
    """

    memcube_id: Optional[str]
    bucket_id: Optional[str]
    blog_markdown: str
    topic: str

    goal_id: Optional[int] = None
    casebook_id: Optional[int] = None
    markdown_path: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BucketEdge:
    src_id: str
    dst_id: str
    edge_type: str  # "similar_to", "cites", "same_doc_section", "related_to"
    weight: float = 1.0


@dataclass
class Bucket:
    topic: str
    nodes: List[BucketNode] = field(default_factory=list)
    edges: List[BucketEdge] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfoSection:
    """
    One section in the eventual Information MemCube / page.
    """

    title: str
    description: str
    content: str
    case_id: Optional[int] = None
    dynamic_scorable_ids: List[int] = field(default_factory=list)


@dataclass
class InformationBuildResult:
    """
    Returned by the orchestrator after a full build.
    """

    topic: str
    target: str
    memcube_id: str
    casebook_id: int
    attributes: Dict[str, Any] = field(default_factory=dict)
    preview_markdown: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasonedSection:
    section_id: str
    title: str
    text: str
    candidates: List[Dict[str, Any]]
    casebook_ref: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class ReasonedBlogResult:
    outline: List[Dict[str, Any]]
    sections: List[ReasonedSection]
    full_text: str
    meta: Dict[str, Any]


@dataclass
class BucketNode:
    """
    A single information fragment collected for a topic.
    Typically corresponds to a paragraph / section / snippet.
    """

    id: str
    source_type: str  # "arxiv_meta", "pdf_section", "wiki", "web", "memcube"
    title: str
    snippet: str
    url: Optional[str] = None
    arxiv_id: Optional[str] = None
    doc_id: Optional[str] = None  # e.g. pdf id, memcube id
    section: Optional[str] = None  # section title / heading
    score: float = 0.0  # initial relevance to topic
    embedding: Optional[list[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SourceProfile:
    name: str
    use_web: bool
    use_arxiv: bool
    use_wikipedia: bool
    use_pdfs: bool
    use_vector_store: bool
    use_internal_memcubes: bool

    max_results_web: int
    max_results_arxiv: int
    max_results_wiki: int
    max_vector_hits: int
    recency_days: int | None



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


@dataclass
class DocumentElement:
    """
    A structured 'visual/structural element' extracted from the PDF.

    This is what your PaddleOCR-VL pass should emit per element.
    """
    id: str                     # stable ID (e.g. paper_id:page:type:index)
    paper_id: str               # arxiv_id or internal paper key
    page: int                   # 1-based page number
    type: ElementType
    bbox: BoundingBox

    # Content payload (only some fields will be used per type)
    text: Optional[str] = None              # for text_block, captions, inline formulas
    latex: Optional[str] = None             # for formula_inline / formula_display
    markdown_table: Optional[str] = None    # for table/chart when converted to md
    image_path: Optional[str] = None        # cropped image written to disk
    caption: Optional[str] = None           # human-ish caption text

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionSpine:
    """
    The 'spine' unit: a logical paper section plus all attached elements.

    This becomes the main unit for the blog generator: one SectionSpine in,
    one blog section out.
    """
    section: "DocumentSection"
    elements: List[DocumentElement] = field(default_factory=list)

    # Optional: convenience fields for downstream logic
    start_page: Optional[int] = None
    end_page: Optional[int] = None



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
