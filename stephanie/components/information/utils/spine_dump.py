# stephanie/components/information/utils/spine_dump.py
from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_snip(text: Optional[str], n: int) -> Optional[str]:
    if not text:
        return None
    t = re.sub(r"\s+", " ", text).strip()
    return t[:n] + ("…" if len(t) > n else "")


def _bbox_to_dict(bbox: Any) -> Optional[Dict[str, float]]:
    if bbox is None:
        return None
    # BoundingBox dataclass (x1,y1,x2,y2) in your codebase
    for k in ("x1", "y1", "x2", "y2"):
        if not hasattr(bbox, k):
            return None
    return {
        "x1": float(getattr(bbox, "x1")),
        "y1": float(getattr(bbox, "y1")),
        "x2": float(getattr(bbox, "x2")),
        "y2": float(getattr(bbox, "y2")),
    }


def _json_default(o: Any):
    # Handles dataclasses + Paths safely
    if isinstance(o, Path):
        return str(o)
    if is_dataclass(o):
        return asdict(o)
    if hasattr(o, "to_dict") and callable(getattr(o, "to_dict")):
        return o.to_dict()
    raise TypeError(f"Not JSON serializable: {type(o)}")


class SpineDumper:
    """
    Writes:
      - spine.json           (compact but readable)
      - spine_graph.json     (nodes/edges for graph viz)
      - spine.dot            (Graphviz DOT)
      - spine.mmd            (Mermaid flowchart)
      - spine_preview.md     (quick human scan)
    """

    def __init__(self, *, run_dir: Path, enabled: bool = True, max_text_chars: int = 240) -> None:
        self.run_dir = Path(run_dir)
        self.enabled = bool(enabled)
        self.max_text_chars = int(max_text_chars)

    def dump(
        self,
        *,
        arxiv_id: str,
        sections: List[Any],
        elements: List[Any],
        spine: List[Any],
        proc_results: Optional[List[Any]] = None,
    ) -> Dict[str, str]:
        if not self.enabled:
            return {}

        self.run_dir.mkdir(parents=True, exist_ok=True)

        payload = self._build_spine_payload(
            arxiv_id=arxiv_id,
            sections=sections,
            elements=elements,
            spine=spine,
            proc_results=proc_results or [],
        )
        graph = self._build_graph(payload)

        out = {}
        out["spine.json"] = self._write_json(self.run_dir / "spine.json", payload)
        out["spine_graph.json"] = self._write_json(self.run_dir / "spine_graph.json", graph)
        out["spine.dot"] = self._write_text(self.run_dir / "spine.dot", self._to_dot(graph))
        out["spine.mmd"] = self._write_text(self.run_dir / "spine.mmd", self._to_mermaid(graph))
        out["spine_preview.md"] = self._write_text(self.run_dir / "spine_preview.md", self._to_markdown(payload))

        return out

    # ---------------------------- builders ----------------------------

    def _build_spine_payload(
        self,
        *,
        arxiv_id: str,
        sections: List[Any],
        elements: List[Any],
        spine: List[Any],
        proc_results: List[Any],
    ) -> Dict[str, Any]:
        # Compact section view
        sec_rows: List[Dict[str, Any]] = []
        for s in sections:
            meta = getattr(s, "metadata", None) or {}
            sec_rows.append(
                {
                    "section_id": getattr(s, "section_id", None),
                    "title": getattr(s, "title", None),
                    "section_index": getattr(s, "section_index", None),
                    "paper_role": getattr(s, "paper_role", None),
                    "start_page": getattr(s, "start_page", meta.get("start_page", None)),
                    "end_page": getattr(s, "end_page", meta.get("end_page", None)),
                    "text_snip": _safe_snip(getattr(s, "text", None), self.max_text_chars),
                    "meta_keys": sorted(list(meta.keys()))[:40],
                }
            )

        # Compact element view
        el_rows: List[Dict[str, Any]] = []
        for e in elements:
            meta = getattr(e, "meta", None) or {}
            el_rows.append(
                {
                    "id": getattr(e, "id", None),
                    "type": getattr(e, "type", None),
                    "page": getattr(e, "page", None),
                    "bbox": _bbox_to_dict(getattr(e, "bbox", None)),
                    "image_path": getattr(e, "image_path", None),
                    "caption_snip": _safe_snip(getattr(e, "caption", None), self.max_text_chars),
                    "text_snip": _safe_snip(getattr(e, "text", None), self.max_text_chars),
                    "meta": meta,
                }
            )

        # Spine mapping (section -> element ids)
        spine_rows: List[Dict[str, Any]] = []
        for ss in spine:
            sec = getattr(ss, "section", None)
            elems = getattr(ss, "elements", None) or []
            spine_rows.append(
                {
                    "section_id": getattr(sec, "section_id", None) if sec else None,
                    "section_title": getattr(sec, "title", None) if sec else None,
                    "start_page": getattr(ss, "start_page", None),
                    "end_page": getattr(ss, "end_page", None),
                    "element_ids": [getattr(x, "id", None) for x in elems],
                    "element_count": len(elems),
                }
            )

        proc_rows: List[Dict[str, Any]] = []
        for r in proc_results or []:
            proc_rows.append(
                {
                    "name": getattr(r, "name", None),
                    "enabled": getattr(r, "enabled", None),
                    "ran": getattr(r, "ran", None),
                    "added_elements": getattr(r, "added_elements", None),
                    "error": getattr(r, "error", None),
                    "stats": getattr(r, "stats", None),
                }
            )

        return {
            "paper": {"arxiv_id": arxiv_id},
            "processors": proc_rows,
            "sections": sec_rows,
            "elements": el_rows,
            "spine": spine_rows,
            "summary": {
                "num_sections": len(sec_rows),
                "num_elements": len(el_rows),
                "num_spine_rows": len(spine_rows),
            },
        }

    def _build_graph(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # section nodes
        for s in payload.get("sections", []):
            sid = s.get("section_id") or f"sec:{s.get('section_index')}"
            nodes.append(
                {
                    "id": sid,
                    "kind": "section",
                    "label": s.get("title") or sid,
                    "section_index": s.get("section_index"),
                }
            )

        # element nodes
        for e in payload.get("elements", []):
            eid = e.get("id") or "elem:unknown"
            label = f"{e.get('type')} p{e.get('page')}"
            nodes.append(
                {
                    "id": eid,
                    "kind": "element",
                    "label": label,
                    "elem_type": e.get("type"),
                    "page": e.get("page"),
                }
            )

        # edges: section -> elements
        for row in payload.get("spine", []):
            sid = row.get("section_id") or f"sec:{row.get('section_title')}"
            for eid in row.get("element_ids", []) or []:
                if not eid:
                    continue
                edges.append({"from": sid, "to": eid, "label": "contains"})

        # edges: section -> next section (by section_index)
        sec_ids = [(s.get("section_index"), s.get("section_id") or f"sec:{s.get('section_index')}") for s in payload.get("sections", [])]
        sec_ids = [(i, sid) for i, sid in sec_ids if i is not None]
        sec_ids.sort(key=lambda x: x[0])
        for (i1, sid1), (i2, sid2) in zip(sec_ids, sec_ids[1:]):
            edges.append({"from": sid1, "to": sid2, "label": "next"})

        return {"nodes": nodes, "edges": edges}

    # ---------------------------- writers ----------------------------

    def _write_json(self, path: Path, obj: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        return str(path)

    def _write_text(self, path: Path, text: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    # ---------------------------- formats ----------------------------

    def _to_dot(self, graph: Dict[str, Any]) -> str:
        lines = ["digraph Spine {", "rankdir=LR;", 'node [shape="box"];']
        for n in graph.get("nodes", []):
            nid = self._dot_escape(n["id"])
            label = self._dot_escape(n.get("label") or n["id"])
            shape = "box" if n.get("kind") == "section" else "ellipse"
            lines.append(f'"{nid}" [label="{label}", shape={shape}];')
        for e in graph.get("edges", []):
            a = self._dot_escape(e["from"])
            b = self._dot_escape(e["to"])
            lab = self._dot_escape(e.get("label", ""))
            lines.append(f'"{a}" -> "{b}" [label="{lab}"];')
        lines.append("}")
        return "\n".join(lines)

    def _to_mermaid(self, graph: Dict[str, Any]) -> str:
        # Mermaid IDs can't like ":" reliably; sanitize but keep a map
        def mid(x: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]", "_", x)

        id_map = {n["id"]: mid(n["id"]) for n in graph.get("nodes", [])}
        lines = ["flowchart LR"]
        for n in graph.get("nodes", []):
            nid = id_map[n["id"]]
            label = (n.get("label") or n["id"]).replace('"', "'")
            if n.get("kind") == "section":
                lines.append(f'{nid}["{label}"]')
            else:
                lines.append(f'{nid}("{label}")')
        for e in graph.get("edges", []):
            a = id_map.get(e["from"], mid(e["from"]))
            b = id_map.get(e["to"], mid(e["to"]))
            lab = e.get("label", "")
            lines.append(f"{a} -->|{lab}| {b}")
        return "\n".join(lines)

    def _to_markdown(self, payload: Dict[str, Any]) -> str:
        paper = payload.get("paper", {})
        lines = [f"# Spine Preview — {paper.get('arxiv_id','unknown')}", ""]
        lines.append(f"- Sections: {payload.get('summary', {}).get('num_sections')}")
        lines.append(f"- Elements: {payload.get('summary', {}).get('num_elements')}")
        lines.append("")

        # Spine list
        for row in payload.get("spine", []):
            title = row.get("section_title") or row.get("section_id")
            lines.append(f"## {title}")
            lines.append(f"- Pages: {row.get('start_page')} → {row.get('end_page')}")
            lines.append(f"- Elements: {row.get('element_count')}")
            for eid in row.get("element_ids", [])[:50]:
                lines.append(f"  - {eid}")
            if (row.get("element_ids") or []) and len(row["element_ids"]) > 50:
                lines.append("  - …")
            lines.append("")
        return "\n".join(lines)

    def _dot_escape(self, s: str) -> str:
        return (s or "").replace("\\", "\\\\").replace('"', '\\"')
