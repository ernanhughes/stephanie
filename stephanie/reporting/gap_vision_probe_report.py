# stephanie/reporting/gap_vision_probe_report.py

"""
Vision-Probe Report
===================

Generates a side-by-side, layout-aware validation report for GAP vision signals.
For each probe graph:
  • Renders VPM frames across selected layouts
  • Computes community separability (layout-dependent)
  • Runs VisionScorer (layout-agnostic) once per probe
  • Produces a Plotly HTML dashboard + JSON metrics

Intended usage:
  from stephanie.reporting.gap_vision_probe_report import GapVisionProbeReport
  GapVisionProbeReport(...).run()

Outputs:
  <out_dir>/vision_probe_report.html
  <out_dir>/vision_probe_metrics.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

from stephanie.components.nexus.graph.graph_layout import render_multi_layout_vpm
from stephanie.services.graph_vision_scorer import VisionScorer

# --------------------------- Helpers ---------------------------

def _mean_pairwise(M: np.ndarray) -> float:
    """Mean pairwise Euclidean distance in a set of points (Nx2)."""
    if len(M) < 2:
        return 0.0
    diff = M[:, None, :] - M[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(len(M), 1)
    return float(d[iu].mean()) if iu[0].size > 0 else 0.0


def community_separability(G: nx.Graph, comms: List[List[int]], positions: Dict) -> float:
    """
    Robust separability for 2+ communities (normalized centroid distance).

    positions: mapping of node-id (as str) -> (x, y). Handles degenerate spans.
    """
    # Normalize positions to [0,1]
    P = {k: (float(v[0]), float(v[1])) for k, v in positions.items()}
    xs = [p[0] for p in P.values()] or [0.0]
    ys = [p[1] for p in P.values()] or [0.0]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = (max_x - min_x) or 1.0
    span_y = (max_y - min_y) or 1.0

    N: Dict[str, Tuple[float, float]] = {}
    for k, (x, y) in P.items():
        nx_ = (x - min_x) / span_x if span_x > 0 else 0.5
        ny_ = (y - min_y) / span_y if span_y > 0 else 0.5
        N[k] = (max(0.0, min(1.0, nx_)), max(0.0, min(1.0, ny_)))

    centroids = []
    intra_dists = []
    for comm in comms:
        pts = np.array([N.get(str(n)) for n in comm if str(n) in N and N.get(str(n)) is not None])
        if len(pts) == 0:
            continue
        centroids.append(pts.mean(axis=0))
        intra_dists.append(_mean_pairwise(pts))

    if len(centroids) < 2:
        return 0.0

    centroids_arr = np.stack(centroids)
    inter_dist = _mean_pairwise(centroids_arr)
    avg_intra = np.mean([d for d in intra_dists if d > 0]) if any(d > 0 for d in intra_dists) else 1e-6
    return inter_dist / avg_intra


def gen_probe(probe_type: str, seed: int = 0) -> Tuple[nx.Graph, List[List[int]]]:
    """Generate standard probe graphs."""
    if probe_type == "sbm":
        blocks = (20, 20, 20)
        p_in, p_out = 0.3, 0.01
        probs = [[p_in if i == j else p_out for j in range(len(blocks))] for i in range(len(blocks))]
        G = nx.stochastic_block_model(blocks, probs, seed=seed)
        comms = [list(range(0, 20)), list(range(20, 40)), list(range(40, 60))]
    elif probe_type == "ring_of_cliques":
        G = nx.ring_of_cliques(4, 8)
        comms = [list(range(i * 8, (i + 1) * 8)) for i in range(4)]
    elif probe_type == "barbell":
        G = nx.barbell_graph(15, 4)
        comms = [list(range(0, 15)), list(range(19, 34))]
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")
    return G, comms


# --------------------------- Report ---------------------------

@dataclass
class GapVisionProbeReport:
    probe_types: List[str] = field(default_factory=lambda: ["sbm", "ring_of_cliques", "barbell"])
    layouts: List[str] = field(default_factory=lambda: ["forceatlas2", "spectral"])
    img_size: int = 256
    model_path: str = "models/graph_vision_scorer.pt"
    output_dir: Path = Path("gap_reports/vision_probes")
    cache_dir: Path = Path(".gap_probe_cache")

    def run(self) -> Dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig = sp.make_subplots(
            rows=len(self.probe_types),
            cols=len(self.layouts) + 1,
            subplot_titles=[f"{probe} - {layout}" for probe in self.probe_types for layout in self.layouts + ["Metrics"]],
            specs=[[{"type": "image"}] * len(self.layouts) + [{"type": "bar"}] for _ in self.probe_types],
            vertical_spacing=0.05,
            horizontal_spacing=0.03,
        )

        scorer = VisionScorer(model_path=self.model_path)
        results = []
        row = 1

        for probe_type in self.probe_types:
            G, comms = gen_probe(probe_type, seed=42)

            # Render once per layout
            vpms, metas = render_multi_layout_vpm(
                G,
                layouts=self.layouts,
                config={"img_size": self.img_size, "cache_dir": str(self.cache_dir)},
            )

            # Vision signals (layout-agnostic) once per probe
            vision_scores = scorer.score_graph(G, cache_key=f"probe_{probe_type}")
            sym_score = float(vision_scores["vision_symmetry"])
            bridge_proxy = float(vision_scores["vision_bridge_proxy"])
            spectral_bucket = vision_scores.get("vision_spectral_gap_bucket")

            row_metrics: Dict[str, Dict[str, float]] = {}
            for col_idx, (layout, meta) in enumerate(zip(self.layouts, metas)):
                # VPM image as-is (uint8 channel 0)
                vpm_img = vpms[col_idx][0]  # [H, W] uint8
                fig.add_trace(go.Image(z=vpm_img), row=row, col=col_idx + 1)

                # Separability
                sep = community_separability(G, comms, meta["positions"])
                row_metrics[layout] = {
                    "separability": float(sep),
                    "spectral_gap": float(meta.get("spectral_gap", 0.0)),
                    "fallback": bool(meta.get("layout_fallback", False)),
                }

                # Inline annotation
                fig.add_annotation(
                    text=f"Sep: {sep:.2f}",
                    x=0.5, y=1.0, xref="paper", yref="paper",
                    xanchor="center", yanchor="bottom",
                    row=row, col=col_idx + 1,
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    bgcolor="rgba(255,255,255,0.85)",
                    borderpad=4,
                )

            # Bar column: per-layout separability + vision signals
            bar_x, bar_y, colors = [], [], []
            for layout in self.layouts:
                bar_x.append(f"{layout}_sep")
                bar_y.append(row_metrics[layout]["separability"])
                colors.append("#1f77b4" if layout == "forceatlas2" else "#9467bd")

            bar_x.extend(["vision_sym", "bridge_proxy"])
            bar_y.extend([sym_score, bridge_proxy])
            colors.extend(["#ff7f0e", "#2ca02c"])

            fig.add_trace(
                go.Bar(
                    x=bar_x,
                    y=bar_y,
                    marker_color=colors,
                    text=[f"{y:.2f}" for y in bar_y],
                    textposition="auto",
                ),
                row=row,
                col=len(self.layouts) + 1,
            )

            results.append({
                "probe_type": probe_type,
                "layouts": row_metrics,
                "vision_scores": {
                    "symmetry": sym_score,
                    "bridge_proxy": bridge_proxy,
                    "spectral_gap_bucket": spectral_bucket,
                },
            })
            row += 1

        # Layout polish
        fig.update_layout(
            title_text="<b>Vision Scorer Validation: Layout Quality vs Structural Signals</b>",
            title_x=0.5,
            height=280 * len(self.probe_types),
            width=1250,
            showlegend=False,
            template="plotly_white",
            font=dict(family="Arial", size=11),
            margin=dict(t=80, l=50, r=50, b=50),
        )
        fig.update_yaxes(title_text="Score", row=len(self.probe_types), col=len(self.layouts) + 1)
        fig.update_xaxes(tickangle=15, row=len(self.probe_types), col=len(self.layouts) + 1)

        # Save
        html_path = self.output_dir / "vision_probe_report.html"
        json_path = self.output_dir / "vision_probe_metrics.json"

        fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            full_html=True,
            config={"displayModeBar": False},
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Console summary
        print(f"✅ Report saved to: {html_path.absolute()}")
        print(f"📊 Metrics saved to: {json_path.absolute()}")
        print("\nKey checks:")
        print("  • FA2 separability > Spectral on ring/barbell (bottlenecks)")
        print("  • Vision symmetry correlates with separability")
        print("  • Barbell: low symmetry (<0.4), high bridge_proxy (>0.6)")
        print("  • SBM: both layouts good; FA2 often edges Spectral")
        print("  • Fallback flag set if FA2 unavailable")

        return {
            "html": str(html_path),
            "json": str(json_path),
            "results": results,
        }

# # scripts/run_gap_vision_probe_report.py
# from __future__ import annotations
# import argparse
# from pathlib import Path
# from stephanie.reporting.gap_vision_probe_report import GapVisionProbeReport

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out", type=str, default="gap_reports/vision_probes")
#     ap.add_argument("--model", type=str, default="models/graph_vision_scorer.pt")
#     ap.add_argument("--img_size", type=int, default=256)
#     ap.add_argument("--probes", type=str, default="sbm,ring_of_cliques,barbell")
#     ap.add_argument("--layouts", type=str, default="forceatlas2,spectral")
#     args = ap.parse_args()

#     report = GapVisionProbeReport(
#         probe_types=[p.strip() for p in args.probes.split(",") if p.strip()],
#         layouts=[l.strip() for l in args.layouts.split(",") if l.strip()],
#         img_size=args.img_size,
#         model_path=args.model,
#         output_dir=Path(args.out),
#     )
#     report.run()

# if __name__ == "__main__":
#     main()
