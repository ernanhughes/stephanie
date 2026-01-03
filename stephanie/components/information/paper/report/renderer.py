# stephanie/components/information/paper/report_renderer.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from stephanie.components.information.data import ConceptCluster, PaperSection


class PaperReportMarkdownRenderer:
    def __init__(
        self,
        *,
        max_sections: int = 16,
        max_clusters: int = 8,
        max_sections_per_cluster: int = 4,
    ):
        self.max_sections = int(max_sections)
        self.max_clusters = int(max_clusters)
        self.max_sections_per_cluster = int(max_sections_per_cluster)

    def render(
        self,
        *,
        arxiv_id: str,
        title: Optional[str],
        graph_stats: Dict[str, Any],
        section_stats: Dict[str, Any],
        cluster_stats: Dict[str, Any],
        match_stats: Dict[str, Any],
        kg_hint: Optional[Dict[str, Any]],
        sections: List[PaperSection],
        clusters: List[ConceptCluster],
    ) -> str:
        lines: List[str] = []

        header = title or f"arXiv {arxiv_id}"
        lines.append("# Paper Knowledge Graph Report")
        lines.append("")
        lines.append(f"**Root paper:** `{arxiv_id}`")
        if title:
            lines.append(f"**Title:** {header}")
        lines.append("")

        # ---- graph overview ----
        lines.append("## Graph Overview")
        lines.append("")
        lines.append(f"- Total papers in graph: **{graph_stats['num_nodes']}**")
        lines.append(f"- Total edges: **{graph_stats['num_edges']}**")
        if graph_stats.get("num_refs") or graph_stats.get("num_similar"):
            lines.append(
                f"- Referenced papers: **{graph_stats['num_refs']}**, "
                f"similar papers: **{graph_stats['num_similar']}**"
            )

        roles = graph_stats.get("roles") or {}
        if roles:
            lines.append("- Roles:")
            for role, count in sorted(roles.items(), key=lambda x: (-x[1], x[0])):
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
        lines.extend(self._render_sections_preview(sections, max_sections=self.max_sections))
        lines.append("")

        # ---- concept clusters ----
        lines.append("## Concept Clusters")
        lines.append("")
        lines.append(f"- Total clusters: **{cluster_stats['total_clusters']}**")
        if cluster_stats.get("cluster_sizes"):
            lines.append("- Cluster sizes: " + ", ".join(str(s) for s in cluster_stats["cluster_sizes"]))
        lines.append("")
        lines.extend(self._render_clusters_preview(sections, clusters, max_clusters=self.max_clusters))
        lines.append("")

        # ---- section matches summary ----
        lines.append("## Cross-Section Links")
        lines.append("")
        lines.append(f"- Similarity matches across sections: **{match_stats['total_matches']}**")
        lines.append("")

        # ---- KG hint ----
        if kg_hint is not None:
            lines.append("## Knowledge Graph Coverage (Nexus)")
            lines.append("")
            lines.append(f"- Sections with KG nodes: **{kg_hint['counted_sections']}**")
            lines.append(f"- Total KG nodes linked to these sections: **{kg_hint['total_nodes']}**")
            lines.append(
                f"- Avg KG nodes per section (where present): "
                f"**{kg_hint['avg_nodes_per_section']:.1f}**"
            )
            lines.append("")

        return "\n".join(lines)

    # --------------------- previews ---------------------------- #

    def _render_sections_preview(self, sections: List[PaperSection], max_sections: int) -> List[str]:
        if not sections:
            return ["_(no sections to preview – pipeline may have failed before section building.)_"]

        lines: List[str] = []
        lines.append("### Sample Sections")
        lines.append("")

        for idx, sec in enumerate(sections[:max_sections]):
            title = (
                getattr(sec, "title", None)
                or getattr(sec, "section_name", None)
                or f"Section {idx + 1}"
            )
            summary = getattr(sec, "summary", None) or getattr(sec, "text", None) or ""
            summary = " ".join(str(summary).split())
            if len(summary) > 220:
                summary = summary[:200] + "..."
            lines.append(f"- **{idx + 1}. {title}** — {summary}")

        if len(sections) > max_sections:
            lines.append(f"_... and {len(sections) - max_sections} more sections._")

        return lines

    def _render_clusters_preview(
        self,
        sections: List[PaperSection],
        clusters: List[ConceptCluster],
        max_clusters: int,
    ) -> List[str]:
        if not clusters:
            return ["_(no clusters to preview.)_"]

        section_index_by_id: Dict[str, int] = {}
        for idx, sec in enumerate(sections):
            sid = getattr(sec, "section_id", None) or getattr(sec, "id", None)
            if sid is not None:
                section_index_by_id[str(sid)] = idx

        lines: List[str] = []
        lines.append("### Sample Concept Clusters")
        lines.append("")

        for i, cl in enumerate(clusters[:max_clusters]):
            label = getattr(cl, "label", None) or getattr(cl, "name", None) or f"Cluster {i + 1}"
            member_ids = (
                getattr(cl, "section_ids", None)
                or getattr(cl, "sections", None)
                or getattr(cl, "members", None)
                or []
            )

            member_indices: List[int] = []
            for mid in member_ids:
                if mid is None:
                    continue
                key = str(getattr(mid, "id", mid))
                if key in section_index_by_id:
                    member_indices.append(section_index_by_id[key])

            member_indices = sorted(set(member_indices))
            if not member_indices:
                lines.append(f"- **{label}** — _no resolvable member sections_")
                continue

            sec_snippets: List[str] = []
            for idx in member_indices[: self.max_sections_per_cluster]:
                sec = sections[idx]
                stitle = getattr(sec, "title", None) or getattr(sec, "section_name", None) or f"S{idx + 1}"
                sec_snippets.append(stitle)

            more = ""
            if len(member_indices) > self.max_sections_per_cluster:
                more = f" (+{len(member_indices) - self.max_sections_per_cluster} more)"

            lines.append(f"- **{label}** — sections: {', '.join(sec_snippets)}{more}")

        if len(clusters) > max_clusters:
            lines.append(f"_... and {len(clusters) - max_clusters} more clusters._")

        return lines
