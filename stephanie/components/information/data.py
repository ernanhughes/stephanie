"""
Core Data Structures for the Information Pipeline.

This module defines the primary data models used throughout the information
retrieval and blog generation pipeline. These structures represent:
- Paper graphs (references, similarity relationships)
- Document sections (chunks of papers)
- Visual elements (figures, tables, formulas)
- Section-element mapping for spine construction

Note: All structures are designed to be serializable to JSON for logging
and persistence in the Nexus knowledge graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from stephanie.components.information_legacy.data import (DocumentElement,
                                                          SectionSpine)

# =============================================================================
# TYPE ALIASES & CONSTANTS
# =============================================================================

ElementType = Literal[
    "figure",
    "table", 
    "formula_inline",
    "formula_display",
    "chart",
    "text_block",
]

PaperRole = Literal[
    "root",              # The seed paper being explained
    "reference",         # Cited by root (direct reference)
    "similar_root",      # Semantically similar to root (from HF)
    "similar_reference", # Semantically similar to a reference
]

EdgeKind = Literal["cites", "similar_to"]



@dataclass
class BlogConfig:
    """
    Portable 'DNA' for a blog-generation run.

    Design goals:
      - JSON-serializable (store in DB, ship to a standalone runner)
      - Backwards compatible (if not provided, defaults behave like baseline)
      - Shared across stages (retrieval/selection/generation/visuals)
    """

    # ---- identity / bookkeeping ----
    version: str = "v1"
    variant: str = "baseline"  # baseline | explore | learned | ablation

    # ---- retrieval / material limits ----
    max_sections: int = 8
    max_reference_items: int = 12
    num_similar_papers: int = 5

    # ---- selection weights (used later by section/paper selectors) ----
    semantic_match_weight: float = 1.0
    domain_match_weight: float = 0.5
    entity_match_weight: float = 0.5

    # ---- generation controls ----
    intro_words: int = 400
    section_words: int = 900
    conclusion_words: int = 400

    # These match your current generator fields (so you can swap in seamlessly)
    intro_model: str = "blog.intro"
    section_model: str = "blog.section"
    conclusion_model: str = "blog.conclusion"

    # ---- visuals/spine ----
    include_visuals: bool = True

    # ---- output ----
    out_root: str = "runs/paper_blogs"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """
        Stable hash for DB indexing / experiment grouping.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_any(cls, obj: Any, *, default: Optional["BlogConfig"] = None) -> "BlogConfig":
        """
        Accept:
          - BlogConfig
          - dict (either full dict, or {"blog_config": {...}})
          - None
        """
        if isinstance(obj, cls):
            return obj
        if obj is None:
            return default or cls()

        if isinstance(obj, dict):
            d = obj.get("blog_config", obj)
            kwargs: Dict[str, Any] = {}
            for name in cls.__dataclass_fields__.keys():
                if name in d:
                    kwargs[name] = d[name]
            # If nothing matched, fall back
            if not kwargs:
                return default or cls()
            return cls(**kwargs)

        return default or cls()


# =============================================================================
# REFERENCE & PAPER GRAPH STRUCTURES
# =============================================================================

@dataclass
class ReferenceRecord:
    """
    Lightweight representation of a reference to an academic paper.
    
    Used as an intermediate representation that can be populated from
    various sources (arXiv API, OpenAlex, PDF parsing, etc.).
    
    Attributes:
        arxiv_id: Primary identifier (e.g., "2401.12345")
        title: Paper title (optional, may be filled later)
        doi: Digital Object Identifier (optional)
        year: Publication year (optional)
        url: Direct link to paper (optional)
        raw_citation: Original citation text (for debugging)
        source: Source of this record ("arxiv", "openalex", "parsed_pdf", etc.)
        raw: Raw response data from the source API (for debugging/extensibility)
    """
    
    arxiv_id: str
    title: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[str] = None
    url: Optional[str] = None
    raw_citation: Optional[str] = None
    source: str = "unknown"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperNode:
    """
    Node in the paper reference graph representing a single academic paper.
    
    Each paper in the retrieval graph (root, references, similar papers)
    becomes a PaperNode. The 'role' field indicates the paper's relationship
    to the seed paper.
    
    Attributes:
        arxiv_id: Primary identifier (e.g., "2401.12345")
        role: Relationship to seed paper (PaperRole)
        title: Paper title
        authors: List of author names
        published_date: Publication date (YYYY-MM-DD format)
        abstract: Paper abstract
        summary: AI-generated summary (optional, filled during processing)
        url: Direct link to paper
        pdf_path: Local path to downloaded PDF (optional)
        text_hash: SHA256 hash of extracted text (for deduplication)
        meta: Additional metadata (conference, venue, etc.)
    """
    
    arxiv_id: str
    role: str  # PaperRole but kept as string for JSON serialization
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    published_date: Optional[str] = None
    abstract: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None

    # Local storage
    pdf_path: Optional[Path] = None
    text_hash: Optional[str] = None
    
    # Extended metadata
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceEdge:
    """
    Directed edge in the paper reference graph.
    
    Represents either citation relationships ("cites") or semantic
    similarity relationships ("similar_to") between papers.
    
    Attributes:
        src: Source paper arXiv ID
        dst: Destination paper arXiv ID  
        kind: Type of relationship (EdgeKind)
        weight: Edge weight (1.0 for citations, similarity score for similar_to)
    """
    
    src: str  # arXiv ID
    dst: str  # arXiv ID
    kind: str  # EdgeKind but kept as string for JSON serialization
    weight: float = 1.0


@dataclass
class PaperReferenceGraph:
    """
    In-memory graph of papers connected by citations and semantic similarity.
    
    This is the primary data structure for the retrieval phase, containing
    the seed paper, its references, and similar papers from embedding search.
    
    Attributes:
        root_arxiv_id: arXiv ID of the seed paper
        nodes: Dictionary mapping arXiv IDs to PaperNode objects
        edges: List of ReferenceEdge objects connecting papers
    """
    
    root_arxiv_id: str
    nodes: Dict[str, PaperNode] = field(default_factory=dict)
    edges: List[ReferenceEdge] = field(default_factory=list)
    
    # -------------------------------------------------------------------------
    # GRAPH OPERATIONS
    # -------------------------------------------------------------------------
    
    def add_node(self, node: PaperNode) -> None:
        """Add a PaperNode to the graph."""
        if not node.arxiv_id:
            raise ValueError("PaperNode requires arxiv_id")
        self.nodes[node.arxiv_id] = node
    
    def get_node(self, arxiv_id: str) -> Optional[PaperNode]:
        """Retrieve a PaperNode by arXiv ID."""
        return self.nodes.get(arxiv_id)
    
    def add_edge(self, src: str, dst: str, kind: str, weight: float = 1.0) -> None:
        """
        Add a directed edge between two papers.
        
        Args:
            src: Source paper arXiv ID
            dst: Destination paper arXiv ID
            kind: Edge type ("cites" or "similar_to")
            weight: Edge weight (default 1.0)
        """
        if src == dst:
            return  # Skip self-loops
        self.edges.append(ReferenceEdge(src=src, dst=dst, kind=kind, weight=weight))
    
    def neighbors(self, arxiv_id: str, kind: Optional[str] = None) -> List[PaperNode]:
        """
        Get all neighboring nodes connected from the given paper.
        
        Args:
            arxiv_id: Source paper arXiv ID
            kind: Optional filter for edge type
            
        Returns:
            List of neighboring PaperNode objects
        """
        out: List[PaperNode] = []
        for edge in self.edges:
            if edge.src == arxiv_id and (kind is None or edge.kind == kind):
                node = self.nodes.get(edge.dst)
                if node:
                    out.append(node)
        return out


# =============================================================================
# DOCUMENT SECTION STRUCTURES
# =============================================================================

@dataclass
class PaperSection:
    """
    A semantically coherent chunk of text from a paper.
    
    Papers are split into sections (typically 1-3 paragraphs each) for
    fine-grained retrieval and processing. Each section can be independently
    summarized, embedded, and scored.
    
    Attributes:
        id: Unique identifier (format: "{paper_arxiv_id}::sec-{section_index}")
        paper_arxiv_id: Source paper arXiv ID
        paper_role: Role of source paper (mirrors PaperNode.role)
        section_index: Sequential index within paper (0-based)
        text: Raw section text
        title: Generated title/summary (optional, filled during processing)
        summary: AI-generated summary (optional)
        start_char: Character offset in original document (optional)
        end_char: Character offset in original document (optional)
        embedding: Vector embedding of section text (optional)
        meta: Additional metadata (section heading, confidence scores, etc.)
    """
    
    id: str
    paper_arxiv_id: str
    paper_role: str
    section_index: int
    
    text: str
    
    title: Optional[str] = None
    summary: Optional[str] = None
    
    # Character offsets for traceability
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Embedding (can be raw vector or reference to embedding store)
    embedding: Any = None
    
    # Extended metadata
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class SectionMatch:
    """
    Directed similarity relationship between two document sections.
    
    Used to connect a section from the seed paper to relevant sections
    from reference/similar papers during content assembly.
    
    Attributes:
        source_section_id: ID of the source (seed) section
        target_section_id: ID of the target (reference) section  
        score: Similarity score (0.0-1.0 or normalized relevance)
        rank: Position in ranked list of matches (1 = best match)
        reason: Human/AI-readable explanation of match (optional)
    """
    
    source_section_id: str
    target_section_id: str
    score: float
    rank: int
    reason: Optional[str] = None


@dataclass
class ConceptCluster:
    """
    Group of sections sharing a common theme or concept.
    
    Produced by clustering algorithms during content organization phase.
    Each cluster becomes a potential section in the final blog post.
    
    Attributes:
        cluster_id: Unique identifier for the cluster
        section_ids: List of DocumentSection IDs in this cluster
        score: Quality/coherence score for the cluster
        label: Human-readable concept label (optional)
        meta: Additional clustering metadata
    """
    
    cluster_id: str
    section_ids: List[str]
    score: float = 1.0
    label: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionNodePayload:
    """
    Minimal payload for inserting a document section into Nexus as a scorable.
    
    This structure is what the Information pipeline produces for consumption
    by the Nexus knowledge graph. All fields are serializable to JSON.
    
    Attributes:
        id: Stable scorable_id (matches DocumentSection.id)
        text: Section text content
        target_type: Nexus target type (e.g., "paper_section")
        paper_id: Source paper arXiv ID
        section_path: Hierarchical section path (e.g., "1/1.2/1.2.3")
        domains: List of domain classifications
        entities: List of extracted named entities
        dims: Dictionary of metric scores (SICQL, HRM, etc.)
        meta: Additional metadata for Nexus
    """
    
    id: str
    text: str
    target_type: str
    paper_id: Optional[str] = None
    section_path: Optional[str] = None
    domains: List[Any] = field(default_factory=list)
    entities: List[Any] = field(default_factory=list)
    dims: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# VISUAL ELEMENT STRUCTURES
# =============================================================================

@dataclass
class BoundingBox:
    """
    2D bounding box in PDF page coordinates.
    
    Coordinates follow PDF convention: origin (0,0) at top-left,
    x increases rightward, y increases downward.
    
    Attributes:
        x1: Left coordinate
        y1: Top coordinate  
        x2: Right coordinate
        y2: Bottom coordinate
    """
    
    x1: float
    y1: float
    x2: float
    y2: float
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------
    
    def height(self) -> float:
        """Return bounding box height."""
        return max(0.0, self.y2 - self.y1)
    
    def width(self) -> float:
        """Return bounding box width."""
        return max(0.0, self.x2 - self.x1)
    
    def center_y(self) -> float:
        """Return vertical center coordinate."""
        return (self.y1 + self.y2) / 2.0
    
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return bounding box as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


# =============================================================================
# SPINE CONSTRUCTION FUNCTIONS
# =============================================================================

def attach_elements_to_sections(
    sections: Iterable[PaperSection],
    elements: Iterable[DocumentElement],
) -> List[SectionSpine]:
    """
    Attach visual elements (figures, tables, formulas) to document sections.
    
    This is the core spine construction algorithm (v1). It uses page-based
    heuristics: elements are attached to sections based on shared page ranges.
    
    Args:
        sections: Iterable of DocumentSection objects
        elements: Iterable of DocumentElement objects (from PaddleOCR-VL)
    
    Returns:
        List of SectionSpine objects in same order as input sections
    
    Note:
        v1 uses simple page range matching. v2 will incorporate vertical
        positioning and semantic relationships for more accurate attachment.
    """
    
    # Materialize inputs (they're often generators)
    section_list = list(sections)
    element_list = list(elements)
    
    # Index elements by page for efficient lookup
    elements_by_page: Dict[int, List[DocumentElement]] = {}
    for element in element_list:
        elements_by_page.setdefault(element.page, []).append(element)
    
    spines: List[SectionSpine] = []
    
    for section in section_list:
        # ---------------------------------------------------------------------
        # DETERMINE PAGE RANGE FOR THIS SECTION
        # ---------------------------------------------------------------------
        start_page: Optional[int] = None
        end_page: Optional[int] = None
        
        # Method 1: Check for explicit page attributes
        if hasattr(section, "start_page") or hasattr(section, "end_page"):
            start_page = getattr(section, "start_page", None)
            end_page = getattr(section, "end_page", None)
            if start_page is not None and end_page is not None:
                start_page = int(start_page)
                end_page = int(end_page)
        
        # Method 2: Check for pages list attribute
        if start_page is None and hasattr(section, "pages"):
            pages = getattr(section, "pages", None) or []
            try:
                page_numbers = list(pages)
                if page_numbers:
                    start_page = int(min(page_numbers))
                    end_page = int(max(page_numbers))
            except (TypeError, ValueError):
                # pages attribute exists but isn't iterable or contains non-ints
                pass
        
        # If we can't determine page range, create spine without elements
        if start_page is None or end_page is None:
            spines.append(SectionSpine(section=section, elements=[]))
            continue
        
        # ---------------------------------------------------------------------
        # COLLECT ELEMENTS IN PAGE RANGE
        # ---------------------------------------------------------------------
        attached_elements: List[DocumentElement] = []
        for page in range(start_page, end_page + 1):
            attached_elements.extend(elements_by_page.get(page, []))
        
        # Create the spine for this section
        spines.append(
            SectionSpine(
                section=section,
                elements=attached_elements,
                start_page=start_page,
                end_page=end_page,
            )
        )
    
    return spines


# =============================================================================
# SERIALIZATION UTILITIES (Optional - could be added later)
# =============================================================================
#
# def to_dict(obj: Any) -> Dict[str, Any]:
#     """Convert dataclass to dictionary, handling Path objects and non-serializable fields."""
#     # Implementation would go here
#     pass
#
# def from_dict(data: Dict[str, Any], cls: Type) -> Any:
#     """Reconstruct dataclass from dictionary."""
#     # Implementation would go here
#     pass