# stephanie/components/information/paper/report_summarizer.py
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
class PaperReportSummary:
    title: Optional[str]
    graph_stats: Dict[str, Any]
    section_stats: Dict[str, Any]
    cluster_stats: Dict[str, Any]
    match_stats: Dict[str, Any]
    kg_hint: Optional[Dict[str, Any]]


class PaperReportSummarizer:
    def __init__(self, *, memory: Any, include_kg_hint: bool = True):
        self.memory = memory
        self.include_kg_hint = include_kg_hint

    def summarize(
        self,
        *,
        arxiv_id: str,
        graph: Optional[PaperReferenceGraph],
        sections: List[PaperSection],
        matches: List[SectionMatch],
        clusters: List[ConceptCluster],
        docs: List[Dict[str, Any]],
    ) -> PaperReportSummary:
        title = self._infer_root_title(arxiv_id, graph, docs)
        graph_stats = self._summarize_graph(graph)
        section_stats = self._summarize_sections(sections)
        cluster_stats = self._summarize_clusters(clusters, sections)
        match_stats = self._summarize_matches(matches)

        kg_hint = None
        if self.include_kg_hint:
            kg_hint = self._summarize_kg_for_sections(sections)

        return PaperReportSummary(
            title=title,
            graph_stats=graph_stats,
            section_stats=section_stats,
            cluster_stats=cluster_stats,
            match_stats=match_stats,
            kg_hint=kg_hint,
        )

    # ----------------------------- title

    def _infer_root_title(
        self,
        arxiv_id: str,
        graph: Optional[PaperReferenceGraph],
        docs: List[Dict[str, Any]],
    ) -> Optional[str]:
        # 1) DocumentStore
        for d in docs:
            if str(d.get("external_id")) == str(arxiv_id):
                t = d.get("title")
                if t:
                    return t

        # 2) graph root node title
        if graph is not None:
            root_id = getattr(graph, "root_id", None)
            nodes = getattr(graph, "nodes", {}) or {}

            if root_id and root_id in nodes:
                node = nodes[root_id]
                t = getattr(node, "title", None)
                if t:
                    return t

            for node in nodes.values():
                role = getattr(node, "role", "") or getattr(node, "kind", "")
                if role == "root":
                    t = getattr(node, "title", None)
                    if t:
                        return t

        return None

    # ----------------------------- summaries

    def _summarize_graph(self, graph: Optional[PaperReferenceGraph]) -> Dict[str, Any]:
        if graph is None:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "num_refs": 0,
                "num_similar": 0,
                "roles": {},
            }

        nodes = getattr(graph, "nodes", {}) or {}
        edges = getattr(graph, "edges", []) or []

        roles: Dict[str, int] = {}
        num_refs = 0
        num_similar = 0

        for node in nodes.values():
            role = getattr(node, "role", None) or getattr(node, "kind", "unknown")
            roles[role] = roles.get(role, 0) 

            if role in ("reference", "ref"):
                num_refs = 1
            elif role in ("similar", "candidate"):
                num_similar = 1

        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "roles": roles,
            "num_refs": num_refs,
            "num_similar": num_similar,
        }

    def _summarize_sections(self, sections: List[PaperSection]) -> Dict[str, Any]:
        total = len(sections)
        lengths: List[int] = []
        for sec in sections:
            txt = getattr(sec, "summary", None) or getattr(sec, "text", "") or ""
            lengths.append(len(txt))

        avg_len = float(sum(lengths) / total) if total > 0 else 0.0
        return {"total_sections": total, "avg_length": avg_len}

    def _summarize_clusters(
        self,
        clusters: List[ConceptCluster],
        sections: List[PaperSection],
    ) -> Dict[str, Any]:
        if not clusters:
            return {"total_clusters": 0, "cluster_sizes": []}

        sizes: List[int] = []
        for cl in clusters:
            sec_ids = (
                getattr(cl, "section_ids", None)
                or getattr(cl, "sections", None)
                or getattr(cl, "members", None)
            )
            sizes.append(len(sec_ids) if isinstance(sec_ids, list) else 0)

        return {"total_clusters": len(clusters), "cluster_sizes": sizes}

    def _summarize_matches(self, matches: List[SectionMatch]) -> Dict[str, Any]:
        if not matches:
            return {"total_matches": 0}
        return {"total_matches": len(matches)}

    def _summarize_kg_for_sections(self, sections: List[PaperSection]) -> Optional[Dict[str, Any]]:
        """
        Optional: rough signal of how much KG/Nexus material we have
        attached to the sections, based on NexusStore.list_nodes_for_scorable.
        """
        nexus = getattr(self.memory, "nexus", None)
        if nexus is None:
            return None

        total_nodes = 0
        counted_sections = 0

        for sec in sections:
            scorable_id = (
                getattr(sec, "scorable_id", None)
                or getattr(sec, "id", None)
                or getattr(sec, "section_id", None)
            )
            if not scorable_id:
                continue
            try:
                nodes = nexus.list_nodes_for_scorable(str(scorable_id), limit=64)
                total_nodes = len(nodes)
                counted_sections = 1
            except Exception:
                continue

        if counted_sections == 0:
            return None

        avg_nodes = total_nodes / max(1, counted_sections)
        return {
            "counted_sections": counted_sections,
            "total_nodes": total_nodes,
            "avg_nodes_per_section": avg_nodes,
        }

