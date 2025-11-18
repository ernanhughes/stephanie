# stephanie/components/nexus/graph/exporters/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    # Avoid import cycles at runtime
    from stephanie.components.nexus.graph.graph import NexusGraph


class BaseGraphExporter(ABC):
    """
    Abstract base for graph exporters.

    Implementations take a NexusGraph (or compatible object) and
    produce artifacts (JSON, HTML, images, etc.).

    Design goals:
      - No ORM leakage: everything goes through NexusGraph
      - Pluggable: you can add more exporters (PyVis, Sigma.js, VPM, etc.)
      - Side-effect focused: methods write to disk and return paths
    """

    def __init__(self, *, name: str = "graph_exporter"):
        self.name = name

    # ---- core JSON payload -------------------------------------------------

    @abstractmethod
    def build_payload(self, graph: "NexusGraph") -> Dict[str, Any]:
        """
        Build a JSON-serialisable payload describing the graph
        for this exporter (e.g., Cytoscape elements).
        """
        raise NotImplementedError

    def write_json(self, graph: "NexusGraph", out_path: Path) -> Path:
        """
        Write the core JSON payload to disk and return the path.
        """
        payload = self.build_payload(graph)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_path

    # ---- HTML view ---------------------------------------------------------

    @abstractmethod
    def write_html(
        self,
        graph: "NexusGraph",
        out_path: Path,
        *,
        title: str = "Nexus graph",
    ) -> Path:
        """
        Write a self-contained HTML page that visualises the graph.
        Implementations can embed the JSON payload directly.
        """
        raise NotImplementedError
