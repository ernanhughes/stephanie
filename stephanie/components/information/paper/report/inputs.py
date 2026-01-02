# stephanie/components/information/paper/report_inputs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from stephanie.components.information.data import (
    ConceptCluster,
    PaperReferenceGraph,
    PaperSection,
    SectionMatch,
)


@dataclass(frozen=True)
class PaperReportInputs:
    arxiv_id: str
    graph: Optional[PaperReferenceGraph]
    sections: List[PaperSection]
    matches: List[SectionMatch]
    clusters: List[ConceptCluster]
    docs: List[Dict[str, Any]]
    graph_json_path: Optional[str]

    @staticmethod
    def from_context(context: Dict[str, Any]) -> PaperReportInputs:
        arxiv_id = context.get("arxiv_id") or context.get("paper_arxiv_id")
        graph: Optional[PaperReferenceGraph] = context.get("paper_graph")
        sections: List[PaperSection] = context.get("paper_sections") or []
        matches: List[SectionMatch] = context.get("section_matches") or []
        clusters: List[ConceptCluster] = context.get("concept_clusters") or []
        docs: List[Dict[str, Any]] = context.get("paper_documents") or []
        graph_json_path = context.get("paper_graph_file")

        return PaperReportInputs(
            arxiv_id=str(arxiv_id) if arxiv_id is not None else "",
            graph=graph,
            sections=sections,
            matches=matches,
            clusters=clusters,
            docs=docs,
            graph_json_path=str(graph_json_path) if graph_json_path else None,
        )

