# stephanie/components/information/paper/report_exporters.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from stephanie.components.information.data import PaperReferenceGraph
from stephanie.components.information.utils.graph_utils import (
    export_graph_pyvis,
    export_paper_graph_json,
)


@dataclass(frozen=True)
class PaperReportArtifacts:
    graph_html_path: Optional[str]
    graph_json_path: Optional[str]
    nexus_tree_json_path: Optional[str]


class PaperReportExporters:
    def __init__(self, *, report_dir: str):
        self.report_dir = report_dir

    def export_graph_artifacts(
        self,
        *,
        graph: Optional[PaperReferenceGraph],
        arxiv_id: str,
        title: str,
    ) -> PaperReportArtifacts:
        graph_html_path: Optional[str] = None
        graph_json_path: Optional[str] = None

        if graph is not None and getattr(graph, "nodes", None):
            graph_html_path = export_graph_pyvis(
                graph=graph,
                arxiv_id=arxiv_id,
                report_dir=self.report_dir,
                title=title,
            )
            graph_json_path = export_paper_graph_json(
                graph=graph,
                arxiv_id=arxiv_id,
                report_dir=self.report_dir,
            )

        return PaperReportArtifacts(
            graph_html_path=graph_html_path,
            graph_json_path=graph_json_path,
            nexus_tree_json_path=None,
        )

    def export_nexus_tree_json(self, local_tree: Any, *, arxiv_id: str) -> str:
        output_path = f"{self.report_dir}/{arxiv_id}_nexus_tree.json"
        payload = self._nexus_tree_to_jsonable(local_tree)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return output_path

    def _nexus_tree_to_jsonable(self, local_tree: Any) -> Dict[str, Any]:
        if hasattr(local_tree, "to_dict") and callable(getattr(local_tree, "to_dict")):
            try:
                data = local_tree.to_dict()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        if hasattr(local_tree, "dict") and callable(getattr(local_tree, "dict")):
            try:
                data = local_tree.dict()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        def _clip(v: Any, max_len: int = 600) -> Any:
            if v is None:
                return None
            if isinstance(v, str):
                return v if len(v) <= max_len else v[:max_len] + "…"
            if isinstance(v, (int, float, bool)):
                return v
            if isinstance(v, list):
                return [_clip(x, max_len=max_len) for x in v[:100]]
            if isinstance(v, dict):
                out: Dict[str, Any] = {}
                for k, val in list(v.items())[:100]:
                    out[str(k)] = _clip(val, max_len=max_len)
                return out
            s = str(v)
            return s if len(s) <= max_len else s[:max_len] + "…"

        nodes_raw = getattr(local_tree, "nodes", None)
        edges_raw = getattr(local_tree, "edges", None)

        if isinstance(nodes_raw, dict):
            nodes_iter = list(nodes_raw.values())
        elif isinstance(nodes_raw, list):
            nodes_iter = nodes_raw
        else:
            nodes_iter = []

        nodes = []
        for n in nodes_iter[:500]:
            if isinstance(n, dict):
                nodes.append(_clip(n))
            else:
                nodes.append(
                    {
                        "id": getattr(n, "id", None) or getattr(n, "node_id", None),
                        "kind": getattr(n, "kind", None) or getattr(n, "node_type", None),
                        "label": getattr(n, "label", None) or getattr(n, "name", None),
                        "meta": _clip(getattr(n, "meta", None) or getattr(n, "attrs", None) or {}),
                    }
                )

        edges_iter = edges_raw if isinstance(edges_raw, list) else []
        edges = []
        for e in edges_iter[:2000]:
            if isinstance(e, dict):
                edges.append(_clip(e))
            else:
                edges.append(
                    {
                        "src": getattr(e, "src", None) or getattr(e, "src_id", None),
                        "dst": getattr(e, "dst", None) or getattr(e, "dst_id", None),
                        "rel": getattr(e, "rel", None) or getattr(e, "relation", None),
                        "weight": getattr(e, "weight", None),
                        "meta": _clip(getattr(e, "meta", None) or {}),
                    }
                )

        if nodes or edges:
            return {"nodes": nodes, "edges": edges, "repr": _clip(repr(local_tree), max_len=400)}

        return {"repr": _clip(repr(local_tree), max_len=2000)}
