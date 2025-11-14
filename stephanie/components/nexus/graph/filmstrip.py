# nexus/graph/filmstrip.py
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


@dataclass
class FrameEvent:
    flash_nodes: Sequence[str] = ()
    flash_edges: Sequence[Tuple[str, str]] = ()
    persist_nodes: Sequence[Tuple[str, Dict]] = ()
    persist_edges: Sequence[Tuple[str, str, Dict]] = ()


class GraphFilmstrip:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.pos_cache: Dict[str, Tuple[float, float]] = {}

    def _stable_pos(self, G: nx.Graph) -> Dict[str, Tuple[float, float]]:
        # reuse previous positions for stability; seed for determinism
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.pos_cache:
            pos = nx.spring_layout(
                G, pos=self.pos_cache, seed=self.seed, k=None, iterations=200
            )
        else:
            pos = nx.spring_layout(G, seed=self.seed, iterations=300)
        self.pos_cache = {n: (float(x), float(y)) for n, (x, y) in pos.items()}
        return self.pos_cache

    def render_frames(
        self,
        baseG: nx.Graph,
        events: List[FrameEvent],
        out_dir: Path,
        fps: int = 2,
        node_attr_quality: str = "quality",
        edge_attr_weight: str = "weight",
    ) -> Tuple[List[Path], Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        frames: List[Path] = []
        G = baseG.copy()

        # baseline frame
        frames.append(self._draw(G, out_dir / "frame_000.png"))

        # event frames
        for i, ev in enumerate(events, start=1):
            # persist additions
            for node_id, attrs in ev.persist_nodes:
                if node_id not in G:
                    G.add_node(node_id, **(attrs or {}))
            for u, v, attrs in ev.persist_edges:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, **(attrs or {}))

            # draw with flashes
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

        # gif
        gif_path = out_dir / "graph_filmstrip.gif"
        imgs = [iio.imread(p) for p in frames]
        iio.mimsave(gif_path, imgs, fps=fps, loop=0)
        return frames, gif_path

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
        flash_nodes = flash_nodes or set()
        flash_edges = flash_edges or set()
        pos = self._stable_pos(G)

        plt.figure(figsize=(8, 8))
        plt.axis("off")
        # node colors/sizes
        qualities = []
        for n, d in G.nodes(data=True):
            q = d.get(node_attr_quality, 0.5)
            try:
                q = float(q)
            except:
                q = 0.5
            qualities.append(q)
        node_sizes = [80 + 220 * q for q in qualities]
        node_colors = qualities  # colormap maps 0..1

        # base edges
        weights = []
        for u, v, d in G.edges(data=True):
            w = d.get(edge_attr_weight, 0.2)
            try:
                w = float(w)
            except:
                w = 0.2
            weights.append(0.5 + 2.5 * w)

        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.35)

        # flashes on top
        if flash_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=list(flash_edges),
                width=3.5,
                alpha=0.85,
                style="solid",
            )

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

        # optional labels for small graphs
        if G.number_of_nodes() <= 60:
            nx.draw_networkx_labels(G, pos, font_size=8)

        plt.tight_layout()
        plt.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        return path
