# stephanie/components/information/utils/spine_dump.py
"""
Module: Spine Dumper

Purpose:
    Exports the document "spine" structure in multiple formats for debugging,
    visualization, and analysis. The spine represents the hierarchical
    organization of a paper's sections and their associated elements
    (figures, tables, equations, etc.) extracted from PDF documents.

Formats Generated:
    - spine.json: Compact JSON representation of the spine structure
    - spine_graph.json: Graph representation (nodes/edges) for visualization
    - spine.dot: Graphviz DOT format for graph rendering
    - spine.mmd: Mermaid flowchart format for documentation
    - spine_preview.md: Human-readable Markdown summary

Key Concepts:
    - Sections: Logical divisions of the paper (Introduction, Methods, etc.)
    - Elements: Atomic components within sections (figures, tables, text blocks)
    - Spine: Mapping between sections and their contained elements

Usage:
    Typically instantiated by paper processing pipelines to export
    intermediate results for debugging and visualization.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.utils.json_sanitize import json_default
from stephanie.utils.text_utils import safe_snip

log = logging.getLogger(__name__)


def _bbox_to_dict(bbox: Any) -> Optional[Dict[str, float]]:
    """
    Convert a bounding box object to a dictionary.
    
    Args:
        bbox: BoundingBox dataclass with x1, y1, x2, y2 attributes
        
    Returns:
        Dictionary with bounding box coordinates or None if invalid
    """
    if bbox is None:
        log.debug("Bounding box is None, returning None")
        return None
    
    # Validate that bbox has required attributes
    for k in ("x1", "y1", "x2", "y2"):
        if not hasattr(bbox, k):
            log.warning(f"Bounding box missing attribute '{k}', returning None")
            return None
    
    result = {
        "x1": float(getattr(bbox, "x1")),
        "y1": float(getattr(bbox, "y1")),
        "x2": float(getattr(bbox, "x2")),
        "y2": float(getattr(bbox, "y2")),
    }
    log.debug(f"Converted bounding box to dict: {result}")
    return result


class SpineDumper:
    """
    Dumps the document spine structure in multiple formats for debugging and visualization.
    
    The spine represents the hierarchical organization of a paper, connecting
    sections (like Introduction, Methods, Results) with their contained elements
    (figures, tables, equations, text blocks). This class exports this structure
    in various formats to aid in debugging the paper-to-blog creation process.
    
    Attributes:
        run_dir: Directory where output files will be written
        enabled: Whether dumping is enabled
        max_text_chars: Maximum characters to include in text snippets
        
    Output Files:
        - spine.json: Complete structured data (primary debug format)
        - spine_graph.json: Graph representation for visualization tools
        - spine.dot: Graphviz DOT format for rendering graphs
        - spine.mmd: Mermaid flowchart for documentation
        - spine_preview.md: Quick human-readable summary
    """
    
    def __init__(
        self, *, run_dir: Path, enabled: bool = True, max_text_chars: int = 240
    ) -> None:
        """
        Initialize the SpineDumper.
        
        Args:
            run_dir: Output directory for dump files
            enabled: Whether dumping is enabled (useful for production vs debug)
            max_text_chars: Maximum characters for text snippets in output
        """
        self.run_dir = Path(run_dir)
        self.enabled = bool(enabled)
        self.max_text_chars = int(max_text_chars)
        
        log.info(f"Initialized SpineDumper: enabled={enabled}, run_dir={run_dir}")
        log.debug(f"Configuration: max_text_chars={max_text_chars}")

    def dump(
        self,
        *,
        arxiv_id: str,
        sections: List[Any],
        elements: List[Any],
        spine: List[Any],
        proc_results: Optional[List[Any]] = None,
    ) -> Dict[str, str]:
        """
        Main entry point: dump the spine structure in multiple formats.
        
        Args:
            arxiv_id: ArXiv identifier for the paper
            sections: List of section objects from document parsing
            elements: List of element objects (figures, tables, etc.)
            spine: List of spine objects mapping sections to elements
            proc_results: Optional list of processor results
            
        Returns:
            Dictionary mapping output filenames to their full paths
            
        Raises:
            IOError: If files cannot be written (logged as ERROR)
        """
        if not self.enabled:
            log.info("Spine dumping is disabled, skipping")
            return {}
        
        log.info(f"Starting spine dump for paper: {arxiv_id}")
        log.debug(f"Input stats: {len(sections)} sections, {len(elements)} elements, "
                 f"{len(spine)} spine mappings")
        
        # Ensure output directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"Output directory ensured: {self.run_dir}")

        # Build the core data structure
        log.debug("Building spine payload...")
        payload = self._build_spine_payload(
            arxiv_id=arxiv_id,
            sections=sections,
            elements=elements,
            spine=spine,
            proc_results=proc_results or [],
        )
        log.debug("Spine payload built successfully")
        
        # Build graph representation
        log.debug("Building graph representation...")
        graph = self._build_graph(payload)
        log.debug(f"Graph built: {len(graph.get('nodes', []))} nodes, "
                 f"{len(graph.get('edges', []))} edges")

        # Generate all output formats
        out = {}
        try:
            log.info("Generating output files...")
            
            out["spine.json"] = self._write_json(
                self.run_dir / "spine.json", payload
            )
            log.info(f"Created spine.json: {out['spine.json']}")
            
            out["spine_graph.json"] = self._write_json(
                self.run_dir / "spine_graph.json", graph
            )
            log.info(f"Created spine_graph.json: {out['spine_graph.json']}")
            
            out["spine.dot"] = self._write_text(
                self.run_dir / "spine.dot", self._to_dot(graph)
            )
            log.info(f"Created spine.dot: {out['spine.dot']}")
            
            out["spine.mmd"] = self._write_text(
                self.run_dir / "spine.mmd", self._to_mermaid(graph)
            )
            log.info(f"Created spine.mmd: {out['spine.mmd']}")
            
            out["spine_preview.md"] = self._write_text(
                self.run_dir / "spine_preview.md", self._to_markdown(payload)
            )
            log.info(f"Created spine_preview.md: {out['spine_preview.md']}")
            
            log.info(f"Spine dump completed successfully for {arxiv_id}")
            log.debug(f"Created {len(out)} output files")
            
        except Exception as e:
            log.error(f"Failed to generate spine dump: {e}", exc_info=True)
            raise
        
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
        """
        Build the main spine data structure.
        
        This creates a compact, serializable representation of the document
        structure, including sections, elements, their relationships, and
        processing metadata.
        
        Returns:
            Dictionary with paper metadata, sections, elements, spine mappings,
            and processing results
        """
        log.debug("Building compact section view...")
        sec_rows: List[Dict[str, Any]] = []
        for i, s in enumerate(sections):
            meta = getattr(s, "metadata", None) or {}
            sec_rows.append(
                {
                    "section_id": getattr(s, "section_id", None),
                    "title": getattr(s, "title", None),
                    "section_index": getattr(s, "section_index", None),
                    "paper_role": getattr(s, "paper_role", None),
                    "start_page": getattr(
                        s, "start_page", meta.get("start_page", None)
                    ),
                    "end_page": getattr(
                        s, "end_page", meta.get("end_page", None)
                    ),
                    "text_snip": safe_snip(
                        getattr(s, "text", None), self.max_text_chars
                    ),
                    "meta_keys": sorted(meta.keys())[:40],
                }
            )
            if i % 10 == 0:  # Log progress every 10 sections
                log.debug(f"Processed {i+1}/{len(sections)} sections")
        
        log.debug(f"Created {len(sec_rows)} section entries")

        log.debug("Building compact element view...")
        el_rows: List[Dict[str, Any]] = []
        for i, e in enumerate(elements):
            meta = getattr(e, "meta", None) or {}
            el_rows.append(
                {
                    "id": getattr(e, "id", None),
                    "type": getattr(e, "type", None),
                    "page": getattr(e, "page", None),
                    "bbox": _bbox_to_dict(getattr(e, "bbox", None)),
                    "image_path": getattr(e, "image_path", None),
                    "caption_snip": safe_snip(
                        getattr(e, "caption", None), self.max_text_chars
                    ),
                    "text_snip": safe_snip(
                        getattr(e, "text", None), self.max_text_chars
                    ),
                    "meta": meta,
                }
            )
            if i % 50 == 0:  # Log progress every 50 elements
                log.debug(f"Processed {i+1}/{len(elements)} elements")
        
        log.debug(f"Created {len(el_rows)} element entries")

        log.debug("Building spine mappings...")
        spine_rows: List[Dict[str, Any]] = []
        for i, ss in enumerate(spine):
            sec = getattr(ss, "section", None)
            elems = getattr(ss, "elements", None) or []
            spine_rows.append(
                {
                    "section_id": getattr(sec, "section_id", None)
                    if sec
                    else None,
                    "section_title": getattr(sec, "title", None)
                    if sec
                    else None,
                    "start_page": getattr(ss, "start_page", None),
                    "end_page": getattr(ss, "end_page", None),
                    "element_ids": [getattr(x, "id", None) for x in elems],
                    "element_count": len(elems),
                }
            )
            if i % 5 == 0:  # Log progress every 5 spine entries
                log.debug(f"Processed {i+1}/{len(spine)} spine mappings")
        
        log.debug(f"Created {len(spine_rows)} spine mappings")

        log.debug("Building processor results...")
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
        
        log.debug(f"Created {len(proc_rows)} processor result entries")

        # Assemble final payload
        payload = {
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
        
        log.debug(f"Final payload summary: {payload['summary']}")
        return payload

    def _build_graph(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert spine payload to graph representation (nodes and edges).
        
        Creates a graph suitable for visualization where:
        - Nodes are sections and elements
        - Edges connect sections to their elements
        - Edges connect sequential sections
        
        Returns:
            Dictionary with 'nodes' and 'edges' lists
        """
        log.debug("Building graph representation from spine payload...")
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # Create section nodes
        log.debug("Creating section nodes...")
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
        log.debug(f"Created {len([n for n in nodes if n['kind'] == 'section'])} section nodes")

        # Create element nodes
        log.debug("Creating element nodes...")
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
        log.debug(f"Created {len([n for n in nodes if n['kind'] == 'element'])} element nodes")

        # Create edges: section -> elements (containment)
        log.debug("Creating containment edges (sections -> elements)...")
        edge_count = 0
        for row in payload.get("spine", []):
            sid = row.get("section_id") or f"sec:{row.get('section_title')}"
            for eid in row.get("element_ids", []) or []:
                if not eid:
                    log.debug(f"Skipping empty element ID for section {sid}")
                    continue
                edges.append({"from": sid, "to": eid, "label": "contains"})
                edge_count += 1
        log.debug(f"Created {edge_count} containment edges")

        # Create edges: section -> next section (sequential flow)
        log.debug("Creating sequential edges (sections -> next sections)...")
        sec_ids = [
            (
                s.get("section_index"),
                s.get("section_id") or f"sec:{s.get('section_index')}",
            )
            for s in payload.get("sections", [])
        ]
        # Filter out sections without index
        sec_ids = [(i, sid) for i, sid in sec_ids if i is not None]
        sec_ids.sort(key=lambda x: x[0])
        log.debug(f"Found {len(sec_ids)} sections with indices for sequencing")
        
        for (i1, sid1), (i2, sid2) in zip(sec_ids, sec_ids[1:]):
            edges.append({"from": sid1, "to": sid2, "label": "next"})
        log.debug(f"Created {len(sec_ids) - 1} sequential edges")

        result = {"nodes": nodes, "edges": edges}
        log.debug(f"Graph built: total {len(nodes)} nodes, {len(edges)} edges")
        return result

    # ---------------------------- writers ----------------------------

    def _write_json(self, path: Path, obj: Any) -> str:
        """
        Write object as JSON file with proper formatting and error handling.
        
        Args:
            path: Output file path
            obj: Python object to serialize
            
        Returns:
            Full path to written file
            
        Raises:
            IOError: If file cannot be written
            TypeError: If object contains non-serializable data
        """
        log.debug(f"Writing JSON to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            json_text = json.dumps(
                obj, indent=2, ensure_ascii=False, default=json_default
            )
            path.write_text(json_text, encoding="utf-8")
            log.debug(f"Successfully wrote {len(json_text)} characters to {path}")
            return str(path)
        except (TypeError, ValueError) as e:
            log.error(f"JSON serialization failed for {path}: {e}")
            raise
        except IOError as e:
            log.error(f"Failed to write JSON file {path}: {e}")
            raise

    def _write_text(self, path: Path, text: str) -> str:
        """
        Write text content to file with error handling.
        
        Args:
            path: Output file path
            text: Text content to write
            
        Returns:
            Full path to written file
            
        Raises:
            IOError: If file cannot be written
        """
        log.debug(f"Writing text file {path} ({len(text)} characters)")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            path.write_text(text, encoding="utf-8")
            log.debug(f"Successfully wrote text file: {path}")
            return str(path)
        except IOError as e:
            log.error(f"Failed to write text file {path}: {e}")
            raise

    # ---------------------------- format converters ----------------------------

    def _to_dot(self, graph: Dict[str, Any]) -> str:
        """
        Convert graph to Graphviz DOT format.
        
        DOT format is used by Graphviz tools to render graphs as images.
        
        Args:
            graph: Graph dictionary with nodes and edges
            
        Returns:
            DOT format string
        """
        log.debug("Converting graph to DOT format...")
        lines = ["digraph Spine {", "rankdir=LR;", 'node [shape="box"];']
        
        # Add nodes
        for n in graph.get("nodes", []):
            nid = self._dot_escape(n["id"])
            label = self._dot_escape(n.get("label") or n["id"])
            shape = "box" if n.get("kind") == "section" else "ellipse"
            lines.append(f'"{nid}" [label="{label}", shape={shape}];')
        
        # Add edges
        for e in graph.get("edges", []):
            a = self._dot_escape(e["from"])
            b = self._dot_escape(e["to"])
            lab = self._dot_escape(e.get("label", ""))
            lines.append(f'"{a}" -> "{b}" [label="{lab}"];')
        
        lines.append("}")
        result = "\n".join(lines)
        log.debug(f"Generated DOT with {len(lines)} lines")
        return result

    def _to_mermaid(self, graph: Dict[str, Any]) -> str:
        """
        Convert graph to Mermaid flowchart syntax.
        
        Mermaid is used for creating flowcharts in Markdown documentation.
        
        Args:
            graph: Graph dictionary with nodes and edges
            
        Returns:
            Mermaid flowchart syntax
        """
        log.debug("Converting graph to Mermaid format...")
        
        def mid(x: str) -> str:
            """Generate Mermaid-safe ID by replacing non-word characters."""
            return re.sub(r"\W", "_", x)

        # Create ID mapping for Mermaid-safe identifiers
        id_map = {n["id"]: mid(n["id"]) for n in graph.get("nodes", [])}
        log.debug(f"Created ID mapping for {len(id_map)} nodes")
        
        lines = ["flowchart LR"]
        
        # Add nodes with appropriate shapes
        for n in graph.get("nodes", []):
            nid = id_map[n["id"]]
            label = (n.get("label") or n["id"]).replace('"', "'")
            if n.get("kind") == "section":
                lines.append(f'{nid}["{label}"]')
            else:
                lines.append(f'{nid}("{label}")')
        
        # Add edges
        for e in graph.get("edges", []):
            a = id_map.get(e["from"], mid(e["from"]))
            b = id_map.get(e["to"], mid(e["to"]))
            lab = e.get("label", "")
            lines.append(f"{a} -->|{lab}| {b}")
        
        result = "\n".join(lines)
        log.debug(f"Generated Mermaid with {len(lines)} lines")
        return result

    def _to_markdown(self, payload: Dict[str, Any]) -> str:
        """
        Create a human-readable Markdown summary of the spine.
        
        This provides a quick overview of the document structure for
        manual inspection.
        
        Args:
            payload: Complete spine data structure
            
        Returns:
            Markdown formatted string
        """
        log.debug("Generating Markdown preview...")
        paper = payload.get("paper", {})
        lines = [f"# Spine Preview — {paper.get('arxiv_id', 'unknown')}", ""]
        
        # Summary statistics
        summary = payload.get("summary", {})
        lines.append(f"- Sections: {summary.get('num_sections')}")
        lines.append(f"- Elements: {summary.get('num_elements')}")
        lines.append(f"- Spine Mappings: {summary.get('num_spine_rows')}")
        lines.append("")

        # Detailed spine listing
        log.debug(f"Including {len(payload.get('spine', []))} spine sections in Markdown")
        for i, row in enumerate(payload.get("spine", [])):
            title = row.get("section_title") or row.get("section_id") or f"Section {i}"
            lines.append(f"## {title}")
            lines.append(
                f"- Pages: {row.get('start_page')} → {row.get('end_page')}"
            )
            lines.append(f"- Elements: {row.get('element_count')}")
            
            # List element IDs (truncated if too many)
            element_ids = row.get("element_ids", [])
            for eid in element_ids[:50]:
                lines.append(f"  - {eid}")
            if element_ids and len(element_ids) > 50:
                lines.append(f"  - … (and {len(element_ids) - 50} more)")
            
            lines.append("")
        
        result = "\n".join(lines)
        log.debug(f"Generated Markdown with {len(lines)} lines")
        return result

    def _dot_escape(self, s: str) -> str:
        """
        Escape special characters for DOT format.
        
        Args:
            s: String to escape
            
        Returns:
            DOT-safe string with backslashes and quotes escaped
        """
        if not s:
            return ""
        escaped = (s or "").replace("\\", "\\\\").replace('"', '\\"')
        log.debug(f"DOT escape: '{s[:50]}...' -> '{escaped[:50]}...'")
        return escaped