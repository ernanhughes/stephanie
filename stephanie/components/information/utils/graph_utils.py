# stephanie/components/information/utils/graph_utils.py
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional

from stephanie.components.information.data import (
    PaperReferenceGraph,
)
from stephanie.components.nexus.graph.exporters.pyvis import export_pyvis_html

log = logging.getLogger(__name__)

def paper_graph_to_jsonable(graph: PaperReferenceGraph) -> Dict[str, Any]:
    """
    Convert a PaperReferenceGraph into a JSON-serializable dict.
    Intended for stable artifact export and downstream visualization tooling.
    """

    def _clip(v: Any, max_len: int = 600) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            return v if len(v) <= max_len else v[:max_len] + "…"
        if isinstance(v, (int, float, bool)):
            return v
        if isinstance(v, list):
            return [_clip(x, max_len=max_len) for x in v[:50]]
        if isinstance(v, dict):
            out: Dict[str, Any] = {}
            for k, val in list(v.items())[:50]:
                out[str(k)] = _clip(val, max_len=max_len)
            return out
        s = str(v)
        return s if len(s) <= max_len else s[:max_len] + "…"

    nodes = []
    for n in (getattr(graph, "nodes", None) or {}).values():
        meta = getattr(n, "meta", None) or {}
        nodes.append(
            {
                "id": getattr(n, "id", None),
                "role": getattr(n, "role", None),
                "title": getattr(n, "title", None),
                "url": getattr(n, "url", None),
                "summary": _clip(getattr(n, "summary", None), max_len=800),
                "meta": _clip(meta, max_len=400),
            }
        )

    edges = []
    for e in (getattr(graph, "edges", None) or []):
        edges.append(
            {
                "src_id": getattr(e, "src_id", None),
                "dst_id": getattr(e, "dst_id", None),
                "rel": getattr(e, "rel", None),
                "weight": getattr(e, "weight", None),
            }
        )

    return {
        "root_id": getattr(graph, "root_id", None),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }

def export_paper_graph_json(
    *,
    graph: PaperReferenceGraph,
    arxiv_id: str,
    report_dir: str,
) -> str:
    """
    Export the paper reference graph as JSON into the per-run report directory.
    Returns the output path.
    """
    output_path = str(Path(report_dir) / f"{arxiv_id}_graph.json")
    payload = paper_graph_to_jsonable(graph)
    try:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info("export_paper_graph_json: saved to %s", output_path)
    except Exception:
        log.exception("export_paper_graph_json: failed to export JSON")
    return output_path


def export_graph_pyvis(
    graph: PaperReferenceGraph,
    arxiv_id: str,
    report_dir: str,
    title: str,
) -> Optional[str]:
    """
    Export the paper graph (PaperReferenceGraph) as a PyVis HTML.

    Uses your existing export_pyvis_html helper and treats graph.nodes as
    the node map and graph.edges as the edge list.
    """
    try:
        nodes = getattr(graph, "nodes", {}) or {}
        edges = list(getattr(graph, "edges", []) or [])

        if not nodes:
            return None

        out_path = Path(report_dir) / f"{arxiv_id}_graph.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        export_pyvis_html(
            nodes=nodes,
            edges=edges,
            output_path=str(out_path),
            title=title,
        )
        log.info(
            "PaperPipelineReportAgent: exported PyVis graph for %s to %s",
            arxiv_id,
            out_path,
        )

        return str(out_path)
    except Exception:
        log.warning(
            "PaperPipelineReportAgent: failed to export PyVis graph for %s",
            arxiv_id,
            exc_info=True,
        )
        return None
