# stephanie/components/information/linkers/overlap.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from stephanie.components.information.data import PaperSection
from stephanie.components.information.graph.paper_graph_abi import GraphEdge
from stephanie.components.information.linkers.base import BaseSectionLinker


class EntityOverlapLinker(BaseSectionLinker):
    """
    Stub: cheap grounding edges via entity overlap.

    Expected future inputs:
      - context["entities_by_section"] = {section_id: ["CLIP","VLM",...]}
    """
    name = "entity_overlap"

    def __init__(self, *, min_jaccard: float = 0.2) -> None:
        self.min_jaccard = float(min_jaccard)

    def link(self, *, root_arxiv_id: str, root_sections: Sequence[PaperSection], corpus_sections: Sequence[PaperSection], context: Dict[str, Any]) -> List[GraphEdge]:
        return []

