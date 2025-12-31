# stephanie/components/information/linkers/knn.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from stephanie.components.information.data import PaperSection
from stephanie.components.information.graph.paper_graph_abi import GraphEdge
from stephanie.components.information.linkers.base import (BaseSectionLinker,
                                                           section_pid)
from stephanie.components.information.tasks.section_link_task import \
    SectionLinkTask


class SemanticKNNLinker(BaseSectionLinker):
    """
    Wraps your existing SectionLinkTask and converts SectionMatch -> GraphEdge.
    """
    name = "semantic_knn"

    def __init__(self, *, top_k: int, min_sim: float, embed_model: Optional[str] = None) -> None:
        self.top_k = int(top_k)
        self.min_sim = float(min_sim)
        self.embed_model = embed_model

    def link(
        self,
        *,
        root_arxiv_id: str,
        root_sections: Sequence[PaperSection],
        corpus_sections: Sequence[PaperSection],
        context: Dict[str, Any],
    ) -> List[GraphEdge]:
        # SectionLinkTask expects a combined list (root + others), with embeddings prepopulated
        all_sections = list(root_sections) + [s for s in corpus_sections if section_pid(s) != root_arxiv_id]

        task = SectionLinkTask(root_arxiv_id=root_arxiv_id, top_k=self.top_k, min_sim=self.min_sim)
        matches, clusters = task.run(all_sections)

        # expose clusters if you want them for reporting
        context["concept_clusters"] = clusters

        edges: List[GraphEdge] = []
        for m in matches:
            edges.append(
                GraphEdge(
                    src=f"section:{m.source_section_id}",
                    dst=f"section:{m.target_section_id}",
                    type="SIMILAR_SECTION",
                    weight=float(m.score),
                    evidence={
                        "rank": int(m.rank),
                        "min_sim": float(self.min_sim),
                        "top_k": int(self.top_k),
                        "embed_model": self.embed_model,
                        "reason": m.reason,
                    },
                )
            )
        return edges
