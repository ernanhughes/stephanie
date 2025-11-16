from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import imageio.v2 as iio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from stephanie.components.nexus.graph.exporters.base import BaseGraphExporter


# --------------------------------------------------------------------------- #
# Frame + filmstrip core                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class FrameEvent:
    """
    A single animation step for the filmstrip.

    - flash_nodes: nodes to highlight in this frame (no structural change)
    - flash_edges: edges to highlight in this frame
    - persist_nodes: nodes to *add* to the base graph for all subsequent frames
    - persist_edges: edges to *add* to the base graph for all subsequent frames
    """
    flash_nodes: Sequence[str] = ()
    flash_edges: Sequence[Tuple[str, str]] = ()
    persist_nodes: Sequence[Tuple[str, Dict]] = ()
    persist_edges: Sequence[Tuple[str, str, Dict]] = ()


class GraphFilmstrip:
    """
    Small helper that turns a NetworkX graph + a sequence of FrameEvents
    into a sequence of PNG frames and a GIF.

    This is intentionally low-level; the exporter will decide how to:
      - construct the base graph
      - choose which nodes/edges to flash
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.pos_cache: Dict[str, Tuple[float, float]] = {}

    # -- layout ------------------------------------------------------------

    def _stable_pos(self, G: nx.Graph) -> Dict[str, Tuple[float, float]]:
        """
        Deterministic spring layout with position reuse so frames don't "jump".
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.pos_cache:
            pos = nx.spring_layout(
                G,
                pos=self.pos_cache,
                seed=self.seed,
                k=None,
                iterations=200,
            )
        else:
            pos = nx.spring_layout(G, seed=self.seed, iterations=300)

        self.pos_cache = {n: (float(x), float(y)) for n, (x, y) in pos.items()}
        return self.pos_cache

    # -- public API --------------------------------------------------------

    def render_frames(
        self,
        baseG: nx.Graph,
        events: List[FrameEvent],
        out_dir: Path,
        fps: int = 2,
        node_attr_quality: str = "quality",
        edge_attr_weight: str = "weight",
    ) -> Tuple[List[Path], Path]:
        """
        Render a list of PNG frames and a GIF.

        Returns:
          (frame_paths, gif_path)
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        frames: List[Path] = []
        G = baseG.copy()

        # Baseline frame: full graph, no highlights
        frames.append(self._draw(
            G,
            out_dir / "frame_000.png",
            node_attr_quality=node_attr_quality,
            edge_attr_weight=edge_attr_weight,
        ))

        # Event frames
        for i, ev in enumerate(events, start=1):
            # Persist additions
            for node_id, attrs in ev.persist_nodes:
                if node_id not in G:
                    G.add_node(node_id, **(attrs or {}))
            for u, v, attrs in ev.persist_edges:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, **(attrs or {}))

            # Draw with flashes
            frames.append(
                self._draw(
                    G,
                    out_dir / f"frame_{i:03d}.png",
                    flash_nodes=set(ev.flash_nodes),
                    flash_edges=set(ev.flash_edges),
                    node_attr_quality=node_attr_quality,
                    edge_attr_weight=edge_attr_weight,
                )
            )

        # Assemble GIF
        gif_path = out_dir / "filmstrip.gif"
        imgs = [iio.imread(p) for p in frames]
        iio.mimsave(gif_path, imgs, fps=fps, loop=0)
        return frames, gif_path

    # -- drawing -----------------------------------------------------------

    def _draw(
        self,
        G: nx.Graph,
        path: Path,
        *,
        flash_nodes: Optional[set] = None,
        flash_edges: Optional[set] = None,
        node_attr_quality: str = "quality",
        edge_attr_weight: str = "weight",
    ) -> Path:
        """
        Draw one frame to disk.
        """
        flash_nodes = flash_nodes or set()
        flash_edges = flash_edges or set()
        pos = self._stable_pos(G)

        plt.figure(figsize=(8, 8))
        plt.axis("off")

        # --- node colors/sizes by "quality" -------------------------------
        qualities: List[float] = []
        for _, d in G.nodes(data=True):
            q = d.get(node_attr_quality, 0.5)
            try:
                q = float(q)
            except Exception:
                q = 0.5
            qualities.append(q)

        node_sizes = [80 + 220 * q for q in qualities]
        node_colors = qualities  # mapped through viridis [0,1]

        # --- base edges ----------------------------------------------------
        weights: List[float] = []
        for _, _, d in G.edges(data=True):
            w = d.get(edge_attr_weight, 0.2)
            try:
                w = float(w)
            except Exception:
                w = 0.2
            weights.append(0.5 + 2.5 * w)

        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.35)

        # --- flash edges ---------------------------------------------------
        if flash_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=list(flash_edges),
                width=3.5,
                alpha=0.85,
                style="solid",
            )

        # --- base nodes ----------------------------------------------------
        cmap = plt.cm.viridis
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            alpha=0.9,
            linewidths=0.0,
        )

        # --- flash nodes ---------------------------------------------------
        if flash_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=list(flash_nodes),
                node_size=260,
                node_color="none",
                edgecolors="white",
                linewidths=2.5,
                alpha=0.95,
            )

        # Optional labels for small graphs
        if G.number_of_nodes() <= 60:
            nx.draw_networkx_labels(G, pos, font_size=8)

        plt.tight_layout()
        plt.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        return path


# --------------------------------------------------------------------------- #
# FilmstripGraphExporter – NexusGraph → GIF filmstrip                         #
# --------------------------------------------------------------------------- #


class FilmstripGraphExporter(BaseGraphExporter):
    """
    Graph-level exporter that turns a NexusGraph into a GIF filmstrip.

    Default behavior:
      - Builds a NetworkX graph with node "quality" and edge "weight".
      - Samples up to `max_events` edges and flashes them one-by-one.
      - Writes:
          - PNG frames into a frames directory
          - filmstrip.gif inside that directory

    Later, you can swap the event construction to use VPMs, pulses, etc.
    """

    def __init__(
        self,
        graph: Any,
        *,
        title: str = "",
        seed: int = 42,
        fps: int = 2,
        node_quality_dim: str = "quality",
        max_events: int = 40,
    ) -> None:
        super().__init__(graph, title=title)
        self.filmstrip = GraphFilmstrip(seed=seed)
        self.fps = int(fps)
        self.node_quality_dim = node_quality_dim
        self.max_events = max_events

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _extract_quality(node: Any, dim: str) -> float:
        """
        Try to find a quality-like metric on a NexusNode / dict.

        Priority:
          - node.dims[dim]
          - node.metrics_vector[dim] / node.vector[dim]
          - node.quality / node["quality"]
        """
        q_val: Optional[float] = None

        # NexusNode.dims
        dims = getattr(node, "dims", None)
        if isinstance(dims, dict) and dim in dims:
            q_val = dims.get(dim)

        # Generic metrics fields on dict-like nodes
        if q_val is None and isinstance(node, dict):
            vec = (
                node.get("metrics_vector")
                or node.get("vector")
                or node.get("dims")
                or {}
            )
            if isinstance(vec, dict):
                q_val = vec.get(dim)

        # Direct quality attr/field
        if q_val is None:
            if hasattr(node, "quality"):
                q_val = getattr(node, "quality")
            elif isinstance(node, dict) and "quality" in node:
                q_val = node["quality"]

        try:
            return float(q_val) if q_val is not None else 0.5
        except Exception:
            return 0.5

    @staticmethod
    def _extract_edge_endpoints(e: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Generic way to read src/dst from ORM or dict.
        """
        def _get(obj: Any, *names: str) -> Optional[Any]:
            for n in names:
                if hasattr(obj, n):
                    v = getattr(obj, n)
                    if v is not None:
                        return v
                if isinstance(obj, dict) and n in obj and obj[n] is not None:
                    return obj[n]
            return None

        s = _get(e, "src", "source", "from")
        t = _get(e, "dst", "target", "to")
        return (str(s) if s is not None else None,
                str(t) if t is not None else None)

    @staticmethod
    def _extract_edge_weight(e: Any) -> float:
        w = None
        if hasattr(e, "weight"):
            w = getattr(e, "weight")
        elif isinstance(e, dict):
            w = e.get("weight")
        try:
            return float(w) if w is not None else 0.2
        except Exception:
            return 0.2

    def _build_base_graph(
        self,
        nodes: Dict[str, Any],
        edges: List[Any],
    ) -> nx.Graph:
        """
        Turn materialized nodes/edges into a NetworkX graph with useful attrs.
        """
        G = nx.Graph()

        # Nodes
        for nid, node in nodes.items():
            quality = self._extract_quality(node, self.node_quality_dim)
            G.add_node(
                str(nid),
                quality=quality,
                # You can add more attributes here (e.g., label, vpm_png, type)
            )

        # Edges
        for e in edges:
            s, t = self._extract_edge_endpoints(e)
            if s is None or t is None:
                continue
            w = self._extract_edge_weight(e)
            G.add_edge(s, t, weight=w)

        return G

    def _build_default_events(
        self,
        G: nx.Graph,
        edges: List[Any],
    ) -> List[FrameEvent]:
        """
        Simple default: sample up to `max_events` edges and flash them.

        This gives a quick, legible "activity sweep" even on large graphs.
        """
        # Map ORM/dict edges to (src, dst)
        edge_pairs: List[Tuple[str, str]] = []
        for e in edges:
            s, t = self._extract_edge_endpoints(e)
            if s is None or t is None:
                continue
            # Ensure the edge actually exists in G
            if G.has_edge(s, t):
                edge_pairs.append((s, t))

        if not edge_pairs:
            return []

        if len(edge_pairs) <= self.max_events:
            selected = edge_pairs
        else:
            # Evenly sample along the list for deterministic coverage
            step = max(1, len(edge_pairs) // self.max_events)
            selected = edge_pairs[::step][: self.max_events]

        events: List[FrameEvent] = []
        for (s, t) in selected:
            events.append(
                FrameEvent(
                    flash_nodes=(s, t),
                    flash_edges=((s, t),),
                )
            )
        return events

    # ---- BaseGraphExporter hook -----------------------------------------

    def export_from_materialized(
        self,
        *,
        nodes: Dict[str, Any],
        edges: List[Any],
        output_path: Path,
        **kwargs,
    ) -> str:
        """
        Render frames + GIF.

        If `output_path` is a directory:
          - frames in that directory
          - GIF as `<output_path>/filmstrip.gif`

        If `output_path` is a .gif path:
          - frames in `<parent>/<stem>_frames/`
          - GIF at the given path (copied/renamed from the frames dir)
        """
        # Decide where frames live
        if output_path.suffix.lower() == ".gif":
            frames_dir = output_path.parent / f"{output_path.stem}_frames"
            gif_target = output_path
        else:
            frames_dir = output_path
            gif_target = frames_dir / "filmstrip.gif"

        frames_dir.mkdir(parents=True, exist_ok=True)

        # Build base graph + events
        G = self._build_base_graph(nodes, edges)
        events = self._build_default_events(G, edges)

        # Render via GraphFilmstrip
        _, gif_path = self.filmstrip.render_frames(
            baseG=G,
            events=events,
            out_dir=frames_dir,
            fps=self.fps,
            node_attr_quality="quality",
            edge_attr_weight="weight",
        )

        # If caller wanted a specific .gif path, honor it
        if gif_target != gif_path:
            gif_target.write_bytes(gif_path.read_bytes())
            return str(gif_target)

        return str(gif_path)


# --------------------------------------------------------------------------- #
# Convenience function                                                        #
# --------------------------------------------------------------------------- #


def export_filmstrip_for_graph(
    graph: Any,
    output_path: Union[str, Path],
    *,
    title: str = "",
    seed: int = 42,
    fps: int = 2,
    node_quality_dim: str = "quality",
    max_events: int = 40,
) -> str:
    """
    Functional wrapper so you can do:

        from ...filmstrip_exporter import export_filmstrip_for_graph

        gif_path = export_filmstrip_for_graph(
            nexus_graph,
            "runs/vpm/8660/filmstrip.gif",
            node_quality_dim="quality",
        )

    Under the hood this uses FilmstripGraphExporter.
    """
    exporter = FilmstripGraphExporter(
        graph,
        title=title,
        seed=seed,
        fps=fps,
        node_quality_dim=node_quality_dim,
        max_events=max_events,
    )
    return exporter.export(output_path)
