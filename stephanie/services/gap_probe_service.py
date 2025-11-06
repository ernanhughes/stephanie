from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from PIL import Image

from stephanie.services.graph_layout import render_multi_layout_vpm


# ---------- Utils ----------
def _normalize_positions(positions: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    xs = [p[0] for p in positions.values()] or [0.0]
    ys = [p[1] for p in positions.values()] or [0.0]
    mnx, mxx = min(xs), max(xs); mny, mxy = min(ys), max(ys)
    sx = (mxx - mnx) or 1.0; sy = (mxy - mny) or 1.0
    return {k: ((v[0]-mnx)/sx, (v[1]-mny)/sy) for k, v in positions.items()}

def _mean_pairwise_dist(coords: np.ndarray) -> float:
    n = len(coords)
    if n < 2: return 0.0
    diffs = coords[:, None, :] - coords[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(n, 1)
    return float(d[iu].mean()) if iu[0].size else 0.0

def community_separability(meta: Dict[str, Any], communities: List[List[Any]]) -> float:
    pos_raw = {k: (float(v[0]), float(v[1])) for k, v in meta["positions"].items()}
    pos = _normalize_positions(pos_raw)
    intra_list, centroids = [], []
    for comm in communities:
        coords = np.array([pos.get(str(n)) for n in comm if str(n) in pos], dtype=float)
        if coords.size == 0:
            intra_list.append(0.0)
            centroids.append(np.zeros(2, dtype=float))
            continue
        intra_list.append(_mean_pairwise_dist(coords))
        centroids.append(coords.mean(axis=0))
    centroids = np.stack(centroids, axis=0) if centroids else np.zeros((0, 2))
    inter = _mean_pairwise_dist(centroids) if len(centroids) > 1 else 0.0
    denom = float(np.mean([x for x in intra_list if x > 0]) if any(x > 0 for x in intra_list) else 1e-6)
    return inter / denom

def to_grid_image(vpm: np.ndarray, scale: int = 1) -> Image.Image:
    C, H, W = vpm.shape
    canv = Image.new("L", (W * C, H), 0)
    for i in range(C):
        ch = Image.fromarray(vpm[i], mode="L")
        canv.paste(ch, (i * W, 0))
    if scale != 1:
        canv = canv.resize((canv.width * scale, canv.height * scale), Image.NEAREST)
    return canv

def save_montage(out_dir: Path, name: str, layouts: List[str], vpms: List[np.ndarray]) -> str:
    rows: List[Image.Image] = []
    for v in vpms:
        strip = to_grid_image(v, scale=2)
        label_h = 12
        row = Image.new("RGB", (strip.width, strip.height + label_h), (0, 0, 0))
        row.paste(Image.merge("RGB", (strip, strip, strip)), (0, label_h))
        rows.append(row)
    H = sum(r.height for r in rows)
    W = max(r.width for r in rows) if rows else 0
    mont = Image.new("RGB", (W, H), (0, 0, 0))
    y = 0
    for r in rows:
        mont.paste(r, (0, y)); y += r.height
    out = out_dir / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    mont.save(out)
    return str(out)

# ---------- Graph builders ----------
def make_sbm(block_sizes=(30, 30, 30), p_in=0.25, p_out=0.02, seed=7):
    probs = [[p_in if i == j else p_out for j in range(len(block_sizes))] for i in range(len(block_sizes))]
    G = nx.stochastic_block_model(block_sizes, probs, seed=seed)
    communities, start = [], 0
    for sz in block_sizes:
        communities.append(list(range(start, start + sz))); start += sz
    return "sbm", G, communities

def make_ring_of_cliques(k=6, size=10):
    G = nx.ring_of_cliques(k, size)
    communities, start = [], 0
    for _ in range(k):
        communities.append(list(range(start, start + size))); start += size
    return "ring_of_cliques", G, communities

def make_barbell(n1=20, n2=20, m=4):
    G = nx.barbell_graph(n1, m, n2)
    left = list(range(0, n1)); right = list(range(n1 + m, n1 + m + n2))
    return "barbell", G, [left, right]

# ---------- Core probe runner ----------
def probe_safe_name(G: nx.Graph) -> str:
    return G.graph.get("name", f"graph_{G.number_of_nodes()}_{G.number_of_edges()}")

def run_probe(G: nx.Graph, communities: List[List[Any]], out_dir: Path, layouts: List[str]) -> Dict[str, Any]:
    vpms, metas = render_multi_layout_vpm(G, layouts=list(layouts))
    name = probe_safe_name(G)
    image_path = save_montage(out_dir, name, layouts, vpms)
    layout_metrics: Dict[str, Dict[str, float]] = {}
    for lay, meta in zip(layouts, metas):
        sep = community_separability(meta, communities)
        layout_metrics[lay] = {
            "separability": float(sep),
            "crossings": float(meta.get("crossings", -1)),
            "spectral_gap": float(meta.get("spectral_gap", 0.0)),
            "n_nodes": float(meta.get("n_nodes", 0)),
            "n_edges": float(meta.get("n_edges", 0)),
        }
    return {"name": name, "image": image_path, "metrics": layout_metrics}

def run_default_suite(out_dir: Path, layouts: List[str] = ("forceatlas2", "spectral")) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for (name, G, comms) in [make_sbm(), make_ring_of_cliques(), make_barbell()]:
        G.graph["name"] = name
        results.append(run_probe(G, comms, out_dir, layouts))
    payload = {"results": results}
    (out_dir / "probe_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
None