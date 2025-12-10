# stephanie/components/information/agents/paper_pipeline_report_agent.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.data import (
    ConceptCluster,
    DocumentSection,
    PaperReferenceGraph,
    SectionMatch,
)


class PaperPipelineReportAgent(BaseAgent):
    """
    Generate a human-readable report from the outputs of PaperPipelineAgent.

    Expects in context:
      - "arxiv_id" or "paper_arxiv_id"
      - "paper_graph": PaperReferenceGraph
      - "paper_sections": List[DocumentSection]
      - "section_matches": List[SectionMatch]
      - "concept_clusters": List[ConceptCluster]
      - optional: "paper_documents": List[dict] from DocumentStore

    Writes:
      - context["paper_report_markdown"]: str
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)

        rcfg = cfg.get("report", {})
        self.max_sections = int(rcfg.get("max_sections", 16))
        self.max_clusters = int(rcfg.get("max_clusters", 8))
        self.max_sections_per_cluster = int(rcfg.get("max_sections_per_cluster", 4))
        self.include_kg_hint = bool(rcfg.get("include_kg_hint", True))

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # ---- inputs from upstream pipeline ----
        arxiv_id = (
            context.get("arxiv_id")
            or context.get("paper_arxiv_id")
            or context.get("root_arxiv_id")
            or "unknown"
        )

        graph: Optional[PaperReferenceGraph] = context.get("paper_graph")
        sections: List[DocumentSection] = context.get("paper_sections") or []
        matches: List[SectionMatch] = context.get("section_matches") or []
        clusters: List[ConceptCluster] = context.get("concept_clusters") or []
        docs: List[Dict[str, Any]] = context.get("paper_documents") or []

        # ---- derive some stats ----
        title = self._infer_root_title(arxiv_id, graph, docs)
        graph_stats = self._summarize_graph(graph)
        section_stats = self._summarize_sections(sections)
        cluster_stats = self._summarize_clusters(clusters, sections)
        match_stats = self._summarize_matches(matches)

        # Optional: light KG/Nexus signal if possible
        kg_hint = None
        if self.include_kg_hint:
            kg_hint = self._summarize_kg_for_sections(sections)

        # ---- build markdown report ----
        md = self._build_markdown_report(
            arxiv_id=arxiv_id,
            title=title,
            graph_stats=graph_stats,
            section_stats=section_stats,
            cluster_stats=cluster_stats,
            match_stats=match_stats,
            kg_hint=kg_hint,
        )

        context["paper_report_markdown"] = md

        # Log a compact event + optionally the report head
        head_preview = md.split("\n", 40)
        head_preview = "\n".join(head_preview[:40])
        if hasattr(self.logger, "log"):
            self.logger.log(
                "PaperPipelineReportGenerated",
                {
                    "arxiv_id": arxiv_id,
                    "root_title": title,
                    "num_sections": section_stats["total_sections"],
                    "num_clusters": cluster_stats["total_clusters"],
                    "num_graph_nodes": graph_stats["num_nodes"],
                    "num_graph_edges": graph_stats["num_edges"],
                },
            )
        else:
            # Fallback to plain logging
            self.logger.info(
                "PaperPipelineReportGenerated arxiv_id=%s\n%s",
                arxiv_id,
                head_preview,
            )

        return context

    # ------------------------------------------------------------------ helpers

    def _infer_root_title(
        self,
        arxiv_id: str,
        graph: Optional[PaperReferenceGraph],
        docs: List[Dict[str, Any]],
    ) -> Optional[str]:
        # 1) try DocumentStore
        for d in docs:
            if str(d.get("external_id")) == str(arxiv_id):
                t = d.get("title")
                if t:
                    return t

        # 2) try graph root node title
        if graph is not None:
            # root node might be stored as graph.root_id or tagged by role
            root_id = getattr(graph, "root_id", None)
            nodes = getattr(graph, "nodes", {}) or {}

            if root_id and root_id in nodes:
                node = nodes[root_id]
                t = getattr(node, "title", None)
                if t:
                    return t

            # otherwise, scan for a node with role "root"
            for node in nodes.values():
                role = getattr(node, "role", "") or getattr(node, "kind", "")
                if role == "root":
                    t = getattr(node, "title", None)
                    if t:
                        return t

        return None

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
            roles[role] = roles.get(role, 0) + 1

            if role in ("reference", "ref"):
                num_refs += 1
            elif role in ("similar", "candidate"):
                num_similar += 1

        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "roles": roles,
            "num_refs": num_refs,
            "num_similar": num_similar,
        }

    def _summarize_sections(
        self, sections: List[DocumentSection]
    ) -> Dict[str, Any]:
        total = len(sections)
        # basic length stats
        lengths = []
        for sec in sections:
            txt = getattr(sec, "summary", None) or getattr(sec, "text", "") or ""
            lengths.append(len(txt))

        avg_len = float(sum(lengths) / total) if total > 0 else 0.0

        return {
            "total_sections": total,
            "avg_length": avg_len,
        }

    def _summarize_clusters(
        self,
        clusters: List[ConceptCluster],
        sections: List[DocumentSection],
    ) -> Dict[str, Any]:
        if not clusters:
            return {"total_clusters": 0, "cluster_sizes": []}

        sizes: List[int] = []
        for cl in clusters:
            # Try a few common patterns for section membership
            sec_ids = (
                getattr(cl, "section_ids", None)
                or getattr(cl, "sections", None)
                or getattr(cl, "members", None)
            )
            if isinstance(sec_ids, list):
                sizes.append(len(sec_ids))
            else:
                sizes.append(0)

        return {
            "total_clusters": len(clusters),
            "cluster_sizes": sizes,
        }

    def _summarize_matches(
        self,
        matches: List[SectionMatch],
    ) -> Dict[str, Any]:
        if not matches:
            return {"total_matches": 0}

        return {"total_matches": len(matches)}

    def _summarize_kg_for_sections(
        self,
        sections: List[DocumentSection],
    ) -> Optional[Dict[str, Any]]:
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
                total_nodes += len(nodes)
                counted_sections += 1
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

    # ------------------------------------------------------------------ markdown

    def _build_markdown_report(
        self,
        *,
        arxiv_id: str,
        title: Optional[str],
        graph_stats: Dict[str, Any],
        section_stats: Dict[str, Any],
        cluster_stats: Dict[str, Any],
        match_stats: Dict[str, Any],
        kg_hint: Optional[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []

        header = title or f"arXiv {arxiv_id}"
        lines.append(f"# Paper Knowledge Graph Report")
        lines.append("")
        lines.append(f"**Root paper:** `{arxiv_id}`")
        if title:
            lines.append(f"**Title:** {title}")
        lines.append("")

        # ---- graph overview ----
        lines.append("## Graph Overview")
        lines.append("")
        lines.append(f"- Total papers in graph: **{graph_stats['num_nodes']}**")
        lines.append(f"- Total edges: **{graph_stats['num_edges']}**")
        if graph_stats["num_refs"] or graph_stats["num_similar"]:
            lines.append(
                f"- Referenced papers: **{graph_stats['num_refs']}**, "
                f"similar papers: **{graph_stats['num_similar']}**"
            )

        if graph_stats["roles"]:
            lines.append("- Roles:")
            for role, count in sorted(
                graph_stats["roles"].items(), key=lambda x: (-x[1], x[0])
            ):
                lines.append(f"  - **{role}**: {count}")
        lines.append("")

        # ---- sections ----
        lines.append("## Sections")
        lines.append("")
        lines.append(
            f"- Total sections: **{section_stats['total_sections']}** "
            f"(avg summary length ≈ {int(section_stats['avg_length'])} chars)"
        )
        lines.append("")

        # We only print a subset to keep report readable
        lines.extend(self._render_sections_preview(max_sections=self.max_sections))
        lines.append("")

        # ---- concept clusters ----
        lines.append("## Concept Clusters")
        lines.append("")
        lines.append(
            f"- Total clusters: **{cluster_stats['total_clusters']}**"
        )
        if cluster_stats["cluster_sizes"]:
            lines.append(
                "- Cluster sizes: "
                + ", ".join(str(s) for s in cluster_stats["cluster_sizes"])
            )
        lines.append("")

        lines.extend(self._render_clusters_preview(max_clusters=self.max_clusters))
        lines.append("")

        # ---- section matches summary ----
        lines.append("## Cross-Section Links")
        lines.append("")
        lines.append(
            f"- Similarity matches across sections: **{match_stats['total_matches']}**"
        )
        lines.append("")

        # ---- KG hint ----
        if kg_hint is not None:
            lines.append("## Knowledge Graph Coverage (Nexus)")
            lines.append("")
            lines.append(
                f"- Sections with KG nodes: **{kg_hint['counted_sections']}**"
            )
            lines.append(
                f"- Total KG nodes linked to these sections: **{kg_hint['total_nodes']}**"
            )
            lines.append(
                f"- Avg KG nodes per section (where present): "
                f"**{kg_hint['avg_nodes_per_section']:.1f}**"
            )
            lines.append("")

        return "\n".join(lines)

    # These two are split out so you can customise them easily if
    # you later want richer previews that use more of your data types.

    def _render_sections_preview(self, max_sections: int) -> List[str]:
        from stephanie.components.information.data import DocumentSection  # local import to avoid cycles

        # We don't have direct access to sections here, so this method is
        # designed to be overridden or adapted if you want richer details.
        # For now, a simple note is returned.
        return [
            "_(section preview omitted – override `_render_sections_preview` "
            "if you want detailed section listings here.)_"
        ]

    def _render_clusters_preview(self, max_clusters: int) -> List[str]:
        return [
            "_(cluster preview omitted – override `_render_clusters_preview` "
            "if you want detailed cluster listings here.)_"
        ]
All right OK