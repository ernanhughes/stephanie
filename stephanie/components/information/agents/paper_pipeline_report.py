# stephanie/components/information/agents/paper_pipeline_report.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.components.information.paper.report_exporters import (
    PaperReportExporters,
)
from stephanie.components.information.paper.report_inputs import (
    PaperReportInputs,
)
from stephanie.components.information.paper.report_renderer import (
    PaperReportMarkdownRenderer,
)
from stephanie.components.information.paper.report_summarizer import (
    PaperReportSummarizer,
)
from stephanie.components.information.paper.report_artifact_writer import (
    ReportArtifactWriter,
)

log = logging.getLogger(__name__)


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
      - context["paper_report_path"]: str
      - context["paper_graph_html_path"]: str (PyVis visualization)
    """

    def __init__(
        self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any
    ):
        super().__init__(
            cfg=cfg, memory=memory, container=container, logger=logger
        )

        rcfg = cfg.get("report", {})
        self.max_sections = int(rcfg.get("max_sections", 16))
        self.max_clusters = int(rcfg.get("max_clusters", 8))
        self.max_sections_per_cluster = int(
            rcfg.get("max_sections_per_cluster", 4)
        )
        self.include_kg_hint = bool(rcfg.get("include_kg_hint", True))
        self.report_dir = rcfg.get(
            "report_dir", f"runs/paper_blogs/{self.run_id}"
        )

        self.write_last = bool(rcfg.get("write_last", True))

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        inputs = PaperReportInputs.from_context(context)
        arxiv_id = inputs.arxiv_id
        graph = inputs.graph
        sections = inputs.sections
        matches = inputs.matches
        clusters = inputs.clusters
        docs = inputs.docs
        graph_json_path = inputs.graph_json_path

        summarizer = PaperReportSummarizer(
            memory=self.memory, include_kg_hint=self.include_kg_hint
        )
        summary = summarizer.summarize(
            arxiv_id=arxiv_id,
            graph=graph,
            sections=sections,
            matches=matches,
            clusters=clusters,
            docs=docs,
        )

        exporters = PaperReportExporters(report_dir=self.report_dir)
        artifacts = exporters.export_graph_artifacts(
            graph=graph,
            arxiv_id=arxiv_id,
            title=summary.title or f"Paper graph {arxiv_id}",
        )
        if artifacts.graph_html_path:
            context["paper_graph_html_path"] = artifacts.graph_html_path
        if artifacts.graph_json_path:
            graph_json_path = artifacts.graph_json_path
            context["paper_graph_json_path"] = graph_json_path

        renderer = PaperReportMarkdownRenderer(
            max_sections=self.max_sections,
            max_clusters=self.max_clusters,
            max_sections_per_cluster=self.max_sections_per_cluster,
        )

        md = renderer.render(
            arxiv_id=arxiv_id,
            title=summary.title,
            graph_stats=summary.graph_stats,
            section_stats=summary.section_stats,
            cluster_stats=summary.cluster_stats,
            match_stats=summary.match_stats,
            kg_hint=summary.kg_hint,
            sections=sections,
            clusters=clusters,
        )

        blog_cfg = (
            context.get("blog_config")
            or context.get("paper_blog_config")
            or (
                self.cfg.get("blog_config")
                if isinstance(self.cfg, dict)
                else None
            )
        )

        writer = ReportArtifactWriter(
            report_dir=self.report_dir, write_last=self.write_last
        )

        if isinstance(blog_cfg, dict) and blog_cfg:
            blog_cfg_out = writer.write_blog_config_snapshot(
                arxiv_id=arxiv_id, blog_cfg=blog_cfg
            )
            context.update(blog_cfg_out)

            # embed in report
            md += "\n\n## Blog config snapshot\n"
            md += (
                f"\n- blog_config_hash: `{context.get('blog_config_hash')}`\n"
            )
            md += (
                "\n```json\n"
                + json.dumps(blog_cfg, indent=2, sort_keys=True)
                + "\n```\n"
            )

        # Log a compact summary
        log.info(
            "PaperPipelineReportGenerated arxiv_id: %s title: %s num_sections: %d num_clusters: %d num_graph_nodes: %d num_graph_edges: %d",
            arxiv_id,
            summary.title,
            summary.section_stats["total_sections"],
            summary.cluster_stats["total_clusters"],
            summary.graph_stats["num_nodes"],
            summary.graph_stats["num_edges"],
        )

        # Export a snapshot of the Nexus local tree (best-effort) so each run is reviewable later.
        nexus_tree_json_path: Optional[str] = None
        try:
            local_tree = self.memory.nexus.log_local_tree(
                run_id=context.get(PIPELINE_RUN_ID),
                root_id=arxiv_id,  # section / scorable id
                depth=self.cfg.get("local_tree_depth", 2),
                max_per_level=self.cfg.get("local_tree_max_per_level", 64),
            )
            nexus_tree_json_path = exporters.export_nexus_tree_json(
                local_tree, arxiv_id=arxiv_id
            )
        except Exception as e:
            log.exception(
                "PaperPipelineReportAgent: failed to export Nexus local tree: %s",
                str(e),
            )

        # Append visualization info if we generated one
        if (
            artifacts.graph_html_path
            or graph_json_path
            or nexus_tree_json_path
        ):
            md += "\n## Visualizations\n\n"
            if artifacts.graph_html_path:
                md += f"- PyVis paper graph: `{artifacts.graph_html_path}`\n"
            if graph_json_path:
                md += f"- Paper graph JSON: `{graph_json_path}`\n"
            if nexus_tree_json_path:
                md += f"- Nexus local tree JSON: `{nexus_tree_json_path}`\n"

        context["paper_report_markdown"] = md

        # ---- save markdown to file (+ optional last copy via last_filename) ----
        write_result = writer.write_report_markdown(arxiv_id=arxiv_id, md=md)
        context["paper_report_path"] = write_result.report_path
        if write_result.report_last_path:
            context["paper_report_path_last"] = write_result.report_last_path

        # Optional convenience: stable 'last_*' artifacts at the report_dir level
        try:
            writer.write_stable_last_artifacts(
                report_path=write_result.report_path,
                graph_html_path=artifacts.graph_html_path,
                graph_json_path=graph_json_path,
                nexus_tree_json_path=nexus_tree_json_path,
            )
        except Exception:
            log.exception(
                "PaperPipelineReportAgent: failed to write last-copy artifacts"
            )

        return context
