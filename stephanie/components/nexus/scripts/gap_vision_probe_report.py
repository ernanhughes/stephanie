# stephanie/components/nexus/scripts/gap_vision_probe_report.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from PIL import Image

from stephanie.components.nexus.graph.graph_layout import render_multi_layout_vpm
from stephanie.services.graph_vision_scorer import VisionScorer

# --------------------------- Config ---------------------------
PROBE_TYPES = ["sbm", "ring_of_cliques", "barbell"]
LAYOUTS = ["forceatlas2", "spectral"]
IMG_SIZE = 256
MODEL_PATH = "models/graph_vision_scorer.pt"  # Train first via your trainer script
OUTPUT_DIR = Path("gap_reports/vision_probes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------- Probe generators (light) ---------------------------
def gen_probe(probe_type: str, seed: int = 0) -> Tuple[nx.Graph, List[List[int]]]:
    """Minimal probe graphs matching your training labels."""
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
        raise ValueError(probe_type)
    return G, comms

# --------------------------- Metrics ---------------------------
def community_separability(G: nx.Graph, comms: List[List[int]], positions: Dict) -> float:
    """Lightweight separability metric (higher = better separated)."""
    coords = []
    for comm in comms:
        pts = np.array([positions[str(n)] for n in comm if str(n) in positions])
        if len(pts) > 0:
            coords.append(pts.mean(axis=0))
    if len(coords) < 2:
        return 0.0
    centroids = np.array(coords)
    intra_dists = []
    for i, comm in enumerate(comms):
        pts = np.array([positions[str(n)] for n in comm if str(n) in positions])
        if len(pts) > 0:
            intra_dists.append(np.mean(np.linalg.norm(pts - centroids[i], axis=1)))
    inter_dist = np.linalg.norm(centroids[0] - centroids[1])
    return inter_dist / (np.mean(intra_dists) + 1e-5)

# --------------------------- Main report ---------------------------
def generate_report():
    # Load vision scorer (cached renders via graph_layout.py)
    scorer = VisionScorer(model_path=MODEL_PATH)

    results = []
    fig = sp.make_subplots(
        rows=len(PROBE_TYPES),
        cols=len(LAYOUTS) + 1,
        subplot_titles=[f"{probe} - {layout}" for probe in PROBE_TYPES for layout in LAYOUTS + ["Metrics"]],
        specs=[[{"type": "image"}] * len(LAYOUTS) + [{"type": "bar"}] for _ in PROBE_TYPES],
        vertical_spacing=0.05,
        horizontal_spacing=0.03,
    )

    row = 1
    all_separability = {layout: [] for layout in LAYOUTS}
    all_symmetry_scores = {layout: [] for layout in LAYOUTS}

    for probe_type in PROBE_TYPES:
        G, comms = gen_probe(probe_type, seed=42)
        vpms, metas = render_multi_layout_vpm(
            G,
            layouts=LAYOUTS,
            config={"img_size": IMG_SIZE, "cache_dir": ".gap_probe_cache"},
        )

        # Compute metrics per layout
        row_metrics = {}
        for col_idx, (layout, meta) in enumerate(zip(LAYOUTS, metas)):
            # 1) Render image for subplot (node density channel)
            vpm_img = vpms[col_idx][0]  # node density channel
            img_pil = Image.fromarray((vpm_img * 255).astype(np.uint8), mode="L")
            
            # 2) Compute separability
            sep = community_separability(G, comms, meta["positions"])
            all_separability[layout].append(sep)
            
            # 3) Run vision scorer
            vision_scores = scorer.score_graph(G, cache_key=f"{probe_type}_{layout}")
            sym_score = vision_scores["vision_symmetry"]
            all_symmetry_scores[layout].append(sym_score)
            
            # Save for JSON
            row_metrics[layout] = {
                "separability": float(sep),
                "vision_symmetry": float(sym_score),
                "spectral_gap": float(meta.get("spectral_gap", 0.0)),
                "bridge_proxy": float(vision_scores["vision_bridge_proxy"]),
            }
            
            # 4) Add image subplot
            fig.add_trace(
                go.Image(z=img_pil),
                row=row,
                col=col_idx + 1
            )
            # Annotate with separability score
            fig.add_annotation(
                x=0.5, y=1.05, xref=f"x{row*3+col_idx+1}", yref=f"y{row*3+col_idx+1}",
                text=f"Sep: {sep:.2f}<br>Sym: {sym_score:.2f}",
                showarrow=False,
                font=dict(size=10, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
            )

        # 5) Add metrics bar chart (separability + symmetry)
        bar_x = LAYOUTS
        bar_y_sep = [row_metrics[lay]["separability"] for lay in LAYOUTS]
        bar_y_sym = [row_metrics[lay]["vision_symmetry"] for lay in LAYOUTS]
        
        fig.add_trace(
            go.Bar(x=bar_x, y=bar_y_sep, name="Separability", marker_color="#1f77b4", opacity=0.7),
            row=row,
            col=len(LAYOUTS) + 1
        )
        fig.add_trace(
            go.Bar(x=bar_x, y=bar_y_sym, name="Vision Symmetry", marker_color="#ff7f0e", opacity=0.7),
            row=row,
            col=len(LAYOUTS) + 1
        )
        
        # Save results
        results.append({
            "probe_type": probe_type,
            "layouts": {layout: row_metrics[layout] for layout in LAYOUTS}
        })
        
        row += 1

    # --------------------------- Global comparison subplot ---------------------------
    fig.update_layout(
        title_text="Vision Scorer Validation: Layout Quality vs Structural Signals",
        title_x=0.5,
        height=300 * len(PROBE_TYPES),
        width=1200,
        showlegend=False,
        template="plotly_white",
        font=dict(family="Arial", size=12),
    )
    fig.update_yaxes(title_text="Score", row=len(PROBE_TYPES), col=len(LAYOUTS)+1)
    fig.update_xaxes(title_text="Layout", row=len(PROBE_TYPES), col=len(LAYOUTS)+1)

    # Save outputs
    html_path = OUTPUT_DIR / "vision_probe_report.html"
    json_path = OUTPUT_DIR / "vision_probe_metrics.json"
    
    fig.write_html(str(html_path))
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Report saved to: {html_path}")
    print(f"📊 Metrics saved to: {json_path}")
    print("\nKey insights to look for:")
    print("- FA2 should show higher separability than Spectral on ring/barbell")
    print("- Vision symmetry score should correlate with separability")
    print("- Barbell should have low symmetry, high bridge proxy")

if __name__ == "__main__":
    generate_report()
