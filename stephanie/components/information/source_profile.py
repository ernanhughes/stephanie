# stephanie/components/information/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BucketNode:
    """
    A single information fragment collected for a topic.
    Typically corresponds to a paragraph / section / snippet.
    """
    id: str
    source_type: str       # "arxiv_meta", "pdf_section", "wiki", "web", "memcube"
    title: str
    snippet: str
    url: Optional[str] = None
    arxiv_id: Optional[str] = None
    doc_id: Optional[str] = None   # e.g. pdf id, memcube id
    section: Optional[str] = None  # section title / heading
    score: float = 0.0             # initial relevance to topic
    embedding: Optional[list[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


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
