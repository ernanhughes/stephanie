# stephanie/components/gap/processors/topology.py
from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
I import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

import numpy as np

logger = logging.getLogger(__name__)

def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _zscore(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    return (M - mu) / (sd + eps)

def plot_topology_holes(run_dir: Path,
                        umap_xy: np.ndarray,
                        X_delta: np.ndarray,
                        H1_bars: list[tuple[float,float]],
                        max_edges: int = 200000) -> dict:
    """
    Visualize holes:
      1) UMAP scatter (density)
      2) Loop overlay using a representative cycle at eps ~ mid of top H1 bar
      3) Graph preview (optional)

    Args
    ----
    run_dir    : base folder for this run (…/base_dir/<run_id>)
    umap_xy    : [N,2] array of UMAP coordinates for the same items
    X_delta    : [N,D] array of Δ-vectors used for PH (SCM-aligned Δ)
    H1_bars    : list of (birth, death) for H1
    max_edges  : guard-rail for dense graphs

    Returns dict with file paths and chosen eps.
    """
    out = {}
    visuals = run_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)

    # (A) UMAP scatter with KDE-like alpha
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(umap_xy[:,0], umap_xy[:,1], s=3, alpha=0.35, linewidths=0)
    ax.set_title("UMAP of Δ-space (space between models)")
    ax.set_xticks([]); ax.set_yticks([])
    umap_png = visuals / "umap_delta_scatter.png"
    fig.savefig(umap_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    out["umap_scatter"] = str(umap_png)

    if not H1_bars:
        out["note"] = "No H1 bars → no loop overlay."
        return out

    # (B) pick the most persistent bar and choose eps = mid(birth, death)
    H1_bars = np.asarray(H1_bars, dtype=float)
    pers = H1_bars[:,1] - H1_bars[:,0]
    top_idx = int(np.argmax(pers))
    b, d = H1_bars[top_idx].tolist()
    eps = 0.5 * (b + d)  # midpoint filtration scale
    out["top_H1_bar"] = {"birth": float(b), "death": float(d), "persistence": float(d-b)}
    out["chosen_eps"] = float(eps)

    # (C) Build eps-graph in Δ-space (threshold graph)
    #     For N up to ~10k this is okay with knn prefilter.
    N = X_delta.shape[0]
    # Get a modest kNN to restrict candidate edges:
    k = min(30, max(10, int(np.sqrt(N))))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X_delta)
    dists, nbrs = nn.kneighbors(X_delta, return_distance=True)

    edges = []
    for i in range(N):
        for dist, j in zip(dists[i], nbrs[i]):
            if i < j and dist <= eps:
                edges.append((i, j, float(dist)))
    # Guard rail
    if len(edges) > max_edges:
        edges = sorted(edges, key=lambda e: e[2])[:max_edges]

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_weighted_edges_from(edges)

    # (D) Find largest component with cycles; extract a cycle basis
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    cycle_nodes = None
    cycle_edges = None
    for comp in comps:
        sub = G.subgraph(comp).copy()
        # Need at least E >= V to have a cycle
        if sub.number_of_edges() >= sub.number_of_nodes():
            # nx.cycle_basis returns list of cycles (each list of nodes)
            cb = nx.cycle_basis(sub)
            if cb:
                # choose the cycle whose perimeter (in Δ-metric) is near eps * len
                # soft heuristic: pick the longest simple cycle for visibility
                cyc = max(cb, key=len)
                cycle_nodes = list(cyc)
                # build edge list in order (close the loop)
                cycle_edges = [(cycle_nodes[t], cycle_nodes[(t+1)%len(cycle_nodes)]) for t in range(len(cycle_nodes))]
                break

    if cycle_nodes is None:
        out["note"] = "No cycle found in eps-graph at chosen eps (try nearby eps)."
        return out

    # (E) Overlay loop on UMAP
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(umap_xy[:,0], umap_xy[:,1], s=2, alpha=0.2, linewidths=0)
    loop_xy = umap_xy[np.array(cycle_nodes)]
    ax.plot(loop_xy[:,0], loop_xy[:,1], lw=2)
    # close loop visually
    ax.plot([loop_xy[-1,0], loop_xy[0,0]], [loop_xy[-1,1], loop_xy[0,1]], lw=2)
    ax.set_title(f"Topological Loop Overlay (ε={eps:.3f})")
    ax.set_xticks([]); ax.set_yticks([])
    loop_png = visuals / "umap_delta_loop_overlay.png"
    fig.savefig(loop_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    out["umap_loop_overlay"] = str(loop_png)

    # (F) Optional: highlight just the component nodes to reduce clutter
    comp_nodes = list(comps[0]) if comps else []
    comp_mask = np.zeros(N, dtype=bool)
    comp_mask[comp_nodes] = True
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(umap_xy[~comp_mask,0], umap_xy[~comp_mask,1], s=2, alpha=0.05, linewidths=0)
    ax.scatter(umap_xy[comp_mask,0], umap_xy[comp_mask,1], s=4, alpha=0.35, linewidths=0)
    ax.plot(loop_xy[:,0], loop_xy[:,1], lw=2)
    ax.plot([loop_xy[-1,0], loop_xy[0,0]], [loop_xy[-1,1], loop_xy[0,1]], lw=2)
    ax.set_title("Component + Representative Loop")
    ax.set_xticks([]); ax.set_yticks([])
    comp_png = visuals / "umap_delta_component_and_loop.png"
    fig.savefig(comp_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    out["umap_component_and_loop"] = str(comp_png)

    return out




def plot_topology_holes_new(run_dir: Path,
                        umap_xy: np.ndarray,
                        X_delta: np.ndarray,
                        H1_bars: list[tuple[float,float]],
                        max_edges: int = 200000) -> dict:
    """
    Visualize holes:
      1) UMAP scatter (density)
      2) Loop overlay using a representative cycle at eps ~ mid of top H1 bar
      3) Graph preview (optional)

    Args
    ----
    run_dir    : base folder for this run (…/base_dir/<run_id>)
    umap_xy    : [N,2] array of UMAP coordinates for the same items
    X_delta    : [N,D] array of Δ-vectors used for PH (SCM-aligned Δ)
    H1_bars    : list of (birth, death) for H1
    max_edges  : guard-rail for dense graphs

    Returns dict with file paths and chosen eps.
    """
    out = {}
    visuals = run_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)

    # (A) UMAP scatter with KDE-like alpha
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(umap_xy[:,0], umap_xy[:,1], s=3, alpha=0.35, linewidths=0)
    ax.set_title("UMAP of Δ-space (space between models)")
    ax.set_xticks([]); ax.set_yticks([])
    umap_png = visuals / "umap_delta_scatter_new.png"
    fig.savefig(umap_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    out["umap_scatter"] = str(umap_png)

    if not H1_bars:
        out["note"] = "No H1 bars → no loop overlay."
        return out

    # (B) pick the most persistent bar and choose eps = mid(birth, death)
    H1_bars = np.asarray(H1_bars, dtype=float)
    pers = H1_bars[:,1] - H1_bars[:,0]
    top_idx = int(np.argmax(pers))
    b, d = H1_bars[top_idx].tolist()
    eps = 0.5 * (b + d)  # midpoint filtration scale
    out["top_H1_bar"] = {"birth": float(b), "death": float(d), "persistence": float(d-b)}
    out["chosen_eps"] = float(eps)

    # (C) Build eps-graph in Δ-space (threshold graph)
    #     For N up to ~10k this is okay with knn prefilter.
    N = X_delta.shape[0]
    # Get a modest kNN to restrict candidate edges:
    k = min(30, max(10, int(np.sqrt(N))))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X_delta)
    dists, nbrs = nn.kneighbors(X_delta, return_distance=True)

    edges = []
    for i in range(N):
        for dist, j in zip(dists[i], nbrs[i]):
            if i < j and dist <= eps:
                edges.append((i, j, float(dist)))
    # Guard rail
    if len(edges) > max_edges:
        edges = sorted(edges, key=lambda e: e[2])[:max_edges]

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_weighted_edges_from(edges)

    # (D) Find largest component with cycles; extract a cycle basis
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    cycle_nodes = None
    cycle_edges = None
    for comp in comps:
        sub = G.subgraph(comp).copy()
        # Need at least E >= V to have a cycle
        if sub.number_of_edges() >= sub.number_of_nodes():
            # nx.cycle_basis returns list of cycles (each list of nodes)
            cb = nx.cycle_basis(sub)
            if cb:
                # choose the cycle whose perimeter (in Δ-metric) is near eps * len
                # soft heuristic: pick the longest simple cycle for visibility
                cyc = max(cb, key=len)
                cycle_nodes = list(cyc)
                # build edge list in order (close the loop)
                cycle_edges = [(cycle_nodes[t], cycle_nodes[(t+1)%len(cycle_nodes)]) for t in range(len(cycle_nodes))]
                break

    if cycle_nodes is None:
        out["note"] = "No cycle found in eps-graph at chosen eps (try nearby eps)."
        return out

    # (E) Overlay loop on UMAP
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(umap_xy[:,0], umap_xy[:,1], s=2, alpha=0.2, linewidths=0)
    loop_xy = umap_xy[np.array(cycle_nodes)]
    ax.plot(loop_xy[:,0], loop_xy[:,1], lw=2)
    # close loop visually
    ax.plot([loop_xy[-1,0], loop_xy[0,0]], [loop_xy[-1,1], loop_xy[0,1]], lw=2)
    ax.set_title(f"Topological Loop Overlay (ε={eps:.3f})")
    ax.set_xticks([]); ax.set_yticks([])
    loop_png = visuals / "umap_delta_loop_overlay.png"
    fig.savefig(loop_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    out["umap_loop_overlay"] = str(loop_png)

    # (F) Optional: highlight just the component nodes to reduce clutter
    comp_nodes = list(comps[0]) if comps else []
    comp_mask = np.zeros(N, dtype=bool)
    comp_mask[comp_nodes] = True
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(umap_xy[~comp_mask,0], umap_xy[~comp_mask,1], s=2, alpha=0.05, linewidths=0)
    ax.scatter(umap_xy[comp_mask,0], umap_xy[comp_mask,1], s=4, alpha=0.35, linewidths=0)
    ax.plot(loop_xy[:,0], loop_xy[:,1], lw=2)
    ax.plot([loop_xy[-1,0], loop_xy[0,0]], [loop_xy[-1,1], loop_xy[0,1]], lw=2)
    ax.set_title("Component + Representative Loop")
    ax.set_xticks([]); ax.set_yticks([])
    comp_png = visuals / "umap_delta_component_and_loop.png"
    fig.savefig(comp_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    out["umap_component_and_loop"] = str(comp_png)

    return out


@dataclass
class TopologyConfig:
    # Which features to use – we’ll default to SCM core-5
    use_weighted: bool = True
    weights: Optional[Dict[str, float]] = None  # keys end-with ".score01" if you use names
    # UMAP/DBSCAN settings (used for visuals only; not for homology)
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.20
    dbscan_eps: float = 0.30
    dbscan_min_samples: int = 5
    # Homology
    max_betti_dim: int = 1
    zscore_inputs: bool = True
    # Stability & nulls
    n_bootstrap: int = 20
    bootstrap_frac: float = 0.8
    n_nulls: int = 50
    random_seed: int = 42

class TopologyProcessor:
    """
    Computes persistent homology on the gap field Δ = H - T (SCM core-5),
    plus stability and null-control analyses. Produces:
      - visuals/pers_diagram_H1.png
      - visuals/pers_barcode_H1.png
      - metrics/betti.json
      - metrics/stability.json
      - metrics/nulls.json
      - (optional) visuals/umap_loop_overlay.png
    """

    def __init__(self, cfg: TopologyConfig, container, logger):
        self.cfg = cfg
        self.container = container
        self.logger = logger

    async def run(self, run_id: str, base_dir: Path) -> Dict[str, Any]:
        base_dir = Path(base_dir)
        vis_dir = base_dir / run_id / "visuals"
        met_dir = base_dir / run_id / "metrics"
        vis_dir.mkdir(parents=True, exist_ok=True)
        met_dir.mkdir(parents=True, exist_ok=True)

        # --- Load SCM-aligned matrices (produced by scoring processor)
        aligned_dir = base_dir / run_id / "aligned"
        H_path = aligned_dir / "hrm_scm_matrix.npy"
        T_path = aligned_dir / "tiny_scm_matrix.npy"
        names_path = aligned_dir / "scm_metric_names.json"

        if not H_path.exists() or not T_path.exists():
            self.logger.log("TopologyMissingMatrices", {
                "run_id": run_id, "hrm_scm_matrix": str(H_path), "tiny_scm_matrix": str(T_path)
            })
            return {"status": "missing_scm_matrices"}

        H = np.load(H_path)
        T = np.load(T_path)

        # Names (we’ll assume Tier-1 are first five entries; else fallback by name)
        if names_path.exists():
            with open(names_path, "r", encoding="utf-8") as f:
                names = json.load(f)
        else:
            # Fallback to first five columns as tier-1
            names = ["scm.reasoning.score01","scm.knowledge.score01","scm.clarity.score01",
                     "scm.faithfulness.score01","scm.coverage.score01",
                     "scm.aggregate01","scm.uncertainty01","scm.ood_hat01","scm.consistency01",
                     "scm.length_norm01","scm.temp01","scm.agree_hat01"]

        # --- Extract SCM core-5 columns by name if available
        col_idx = []
        want = ["scm.reasoning.score01","scm.knowledge.score01","scm.clarity.score01",
                "scm.faithfulness.score01","scm.coverage.score01"]
        name_to_idx = {n:i for i,n in enumerate(names)}
        for w in want:
            if w in name_to_idx:
                col_idx.append(name_to_idx[w])
        if len(col_idx) != 5:  # fallback: first 5 columns
            col_idx = list(range(min(5, H.shape[1])))

        H5 = H[:, col_idx]
        T5 = T[:, col_idx]

        if H5.shape != T5.shape or H5.shape[1] == 0:
            self.logger.log("TopologyShapeMismatch", {"H5": H5.shape, "T5": T5.shape})
            return {"status": "shape_mismatch"}

        # --- Δ cloud (optionally zscore each model first)
        if self.cfg.zscore_inputs:
            H5 = _zscore(H5)
            T5 = _zscore(T5)
        Delta = H5 - T5   # (N, 5)

        # Optional: column weights
        if self.cfg.use_weighted and self.cfg.weights:
            # Map weights by suffix name if possible; else uniform
            W = np.ones(Delta.shape[1], dtype=np.float32)
            for i, n in enumerate([names[j] for j in col_idx]):
                # Find a weight for e.g. "reasoning.score01" (suffix after "scm.")
                suf = n.split("scm.")[-1]
                W[i] = float(self.cfg.weights.get(suf, 1.0))
            Delta = Delta * W[None, :]

        # --- Persistent homology on Δ
        betti = self._compute_ph_and_figures(Delta, vis_dir)
        _save_json(met_dir / "betti.json", betti)

        # --- Stability: bootstraps + weight jitter
        stability = self._stability_checks(Delta, names=[names[j] for j in col_idx], vis_dir=vis_dir)
        _save_json(met_dir / "stability.json", stability)

        # --- Null controls
        nulls = self._null_controls(H5, T5)
        _save_json(met_dir / "nulls.json", nulls)

        # Optional: UMAP for storytelling (not used for homology)
        U = None
        try:
            umap_png = vis_dir / "umap_loop_overlay.png"
            U = self._umap_overlay(Delta, out_png=umap_png)
        except Exception as e:
            logger.info(f"[Topology] UMAP overlay skipped: {e}")

        # If we have UMAP and H1 bars, draw the representative loop on top of UMAP
        try:
            if U is not None and betti.get("H1_bars"):
                loop_vis = plot_topology_holes(
                    run_dir=base_dir / run_id,
                    umap_xy=U,
                    X_delta=Delta,
                    H1_bars=[tuple(b) for b in betti["H1_bars"]],
                    max_edges=200000
                )
                # Optional: log paths for your report
                self.logger.log("TopologyLoopOverlay", loop_vis)
        except Exception as e:
            logger.info(f"[Topology] Loop overlay skipped: {e}")

        # --- Validate assumptions early
        tda_assumptions = self._validate_tda_assumptions(Delta)
        _save_json(met_dir / "tda_assumptions.json", tda_assumptions)

        # --- Persistent homology
        betti = self._compute_ph_and_figures(Delta, vis_dir)
        _save_json(met_dir / "betti.json", betti)

        # --- Stability (bootstrap + weight jitter)
        stability = self._stability_checks(Delta, names=[names[j] for j in col_idx], vis_dir=vis_dir)
        self.stability_results = stability  # keep for significance
        _save_json(met_dir / "stability.json", stability)

        # --- Null controls
        nulls = self._null_controls(H5, T5)
        _save_json(met_dir / "nulls.json", nulls)

        # --- Optional: UMAP + loop overlay (already in your file)
        U = None
        try:
            umap_png = vis_dir / "umap_loop_overlay.png"
            U = self._umap_overlay(Delta, out_png=umap_png)
        except Exception as e:
            logger.info(f"[Topology] UMAP overlay skipped: {e}")
        try:
            if U is not None and betti.get("H1_bars"):
                loop_vis = plot_topology_holes(
                    run_dir=base_dir / run_id,
                    umap_xy=U,
                    X_delta=Delta,
                    H1_bars=[tuple(b) for b in betti["H1_bars"]],
                    max_edges=200000
                )
                self.logger.log("TopologyLoopOverlay", loop_vis)
        except Exception as e:
            logger.info(f"[Topology] Loop overlay skipped: {e}")

        # --- Formal significance & corrections (only if there ARE loops)
        if betti.get("b1", 0) > 0:
            significance = self._statistical_significance(betti["top_H1_persistence"], nulls)
            _save_json(met_dir / "statistical_significance.json", significance)

            mult_test = self._correct_multiple_testing(betti.get("H1_bars", []), nulls)
            _save_json(met_dir / "multiple_testing_correction.json", mult_test)

            param_sens = self._parameter_sensitivity_analysis(Delta, betti["top_H1_persistence"])
            _save_json(met_dir / "parameter_sensitivity.json", param_sens)
        else:
            significance, mult_test, param_sens = (
                {"p_value": 1.0, "significance": "no_h1_bars"},
                {"total_bars": 0, "significant_bars_bh_5pct": 0, "q_values": []},
                {},
            )

        out = {
            "status": "ok",
            "betti": betti,
            "stability": stability,
            "nulls": nulls,
            "statistical_significance": significance,
            "multiple_testing": mult_test,
            "parameter_sensitivity": param_sens,
            "tda_assumptions": tda_assumptions,
            "figures": {
                "H1_persistence_diagram": str((vis_dir / "pers_diagram_H1.png").resolve()),
                "H1_barcode": str((vis_dir / "pers_barcode_H1.png").resolve()),
                "umap_overlay": str((vis_dir / "umap_loop_overlay.png").resolve()),
                "umap_xy_npy": str((vis_dir / "umap_loop_overlay.npy").resolve()),
            },
        }
        return out

    # ---------------------------------------------------------------------
    # Core PH
    # ---------------------------------------------------------------------
    def _compute_ph_and_figures(self, Delta: np.ndarray, vis_dir: Path) -> Dict[str, Any]:
        from ripser import ripser
        import matplotlib.pyplot as plt
        from persim import plot_diagrams

        res = ripser(Delta, maxdim=self.cfg.max_betti_dim)
        dgms = res["dgms"]

        # Save diagrams
        try:
            plt.figure(figsize=(6, 5))
            plot_diagrams(dgms, show=False)
            plt.title("Persistence Diagrams (H0/H1)")
            (vis_dir / "pers_diagram_all.png").parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(vis_dir / "pers_diagram_all.png", dpi=160, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        # H1 only
        H1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
        try:
            plt.figure(figsize=(6, 5))
            plot_diagrams([H1], show=False)
            plt.title("Persistence Diagram (H1)")
            plt.savefig(vis_dir / "pers_diagram_H1.png", dpi=160, bbox_inches="tight")
            plt.close()

            # Barcode
            if H1.shape[0] > 0:
                lives = H1[:, 1] - H1[:, 0]
                order = np.argsort(lives)[::-1]
                H1_sorted = H1[order]
                # Simple barcode plot
                plt.figure(figsize=(8, max(3, 0.25 * len(H1_sorted))))
                for i, (b, d) in enumerate(H1_sorted):
                    plt.hlines(y=i, xmin=b, xmax=d, linewidth=2)
                plt.xlabel("Filtration scale")
                plt.ylabel("H1 features (sorted by persistence)")
                plt.title("Persistence Barcode (H1)")
                plt.savefig(vis_dir / "pers_barcode_H1.png", dpi=160, bbox_inches="tight")
                plt.close()
        except Exception:
            pass

        b0 = int(dgms[0].shape[0]) if len(dgms) > 0 else 0
        b1 = int(H1.shape[0])
        top_H1 = 0.0 if b1 == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
        return {
            "b0": b0,
            "b1": b1,
            "top_H1_persistence": top_H1,
            "H1_bars": [[float(a), float(b)] for a, b in (H1 if b1 else [])],
        }

    # ---------------------------------------------------------------------
    # Stability
    # ---------------------------------------------------------------------
    def _stability_checks(self, Delta: np.ndarray, names: list[str], vis_dir: Path) -> Dict[str, Any]:
        from ripser import ripser

        N = Delta.shape[0]
        bstrap = []
        for _ in range(max(1, int(self.cfg.n_bootstrap))):
            m = max(2, int(self.cfg.bootstrap_frac * N))
            idx = np.random.choice(N, size=m, replace=False)
            dsub = Delta[idx]
            dgms = ripser(dsub, maxdim=self.cfg.max_betti_dim)["dgms"]
            H1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
            top = 0.0 if H1.shape[0] == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
            bstrap.append({"b1": int(H1.shape[0]), "top_H1_persistence": top})

        # Weight jitter (±10% if using weights)
        jitter = []
        if self.cfg.use_weighted:
            W = np.ones(Delta.shape[1], dtype=np.float32)
            for _ in range(10):
                j = 1.0 + 0.1 * (2 * np.random.rand(Delta.shape[1]) - 1)  # ±10%
                dJ = Delta * j[None, :]
                dgms = ripser(dJ, maxdim=self.cfg.max_betti_dim)["dgms"]
                H1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
                top = 0.0 if H1.shape[0] == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
                jitter.append({"b1": int(H1.shape[0]), "top_H1_persistence": top})

        return {"bootstrap": bstrap, "weight_jitter": jitter}

    # ---------------------------------------------------------------------
    # Nulls
    # ---------------------------------------------------------------------
    def _null_controls(self, H5: np.ndarray, T5: np.ndarray) -> Dict[str, Any]:
        from ripser import ripser
        N = H5.shape[0]

        # A) Shuffled pairing H - T_pi
        shuffled = []
        for _ in range(max(5, int(self.cfg.n_nulls))):
            perm = np.random.permutation(N)
            d = H5 - T5[perm]
            dgms = ripser(d, maxdim=self.cfg.max_betti_dim)["dgms"]
            H1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
            top = 0.0 if H1.shape[0] == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
            shuffled.append({"b1": int(H1.shape[0]), "top_H1_persistence": top})

        # B) Self-difference H - H (should be trivial)
        selfH = ripser(H5 - H5, maxdim=self.cfg.max_betti_dim)["dgms"]
        H1_selfH = selfH[1] if len(selfH) > 1 else np.zeros((0, 2))
        selfH_top = 0.0 if H1_selfH.shape[0] == 0 else float(np.max(H1_selfH[:, 1] - H1_selfH[:, 0]))

        selfT = ripser(T5 - T5, maxdim=self.cfg.max_betti_dim)["dgms"]
        H1_selfT = selfT[1] if len(selfT) > 1 else np.zeros((0, 2))
        selfT_top = 0.0 if H1_selfT.shape[0] == 0 else float(np.max(H1_selfT[:, 1] - H1_selfT[:, 0]))

        return {
            "shuffled_pairing": shuffled,
            "self_diff": {"H_minus_H_top": selfH_top, "T_minus_T_top": selfT_top},
        }

    # ---------------------------------------------------------------------
    # UMAP – for storytelling (not used in homology)
    # ---------------------------------------------------------------------
    def _umap_overlay(self, Delta: np.ndarray, out_png: Path) -> np.ndarray:
        import umap
        import matplotlib.pyplot as plt
        from sklearn.cluster import DBSCAN

        reducer = umap.UMAP(
            n_neighbors=self.cfg.umap_n_neighbors,
            min_dist=self.cfg.umap_min_dist,
            metric="euclidean",
            random_state=42,
        )
        U = reducer.fit_transform(Delta)  # (N, 2)

        # Simple DBSCAN to outline dense regions
        cl = DBSCAN(eps=self.cfg.dbscan_eps, min_samples=self.cfg.dbscan_min_samples)
        labels = cl.fit_predict(U)

        plt.figure(figsize=(7, 6))
        K = max(labels) + 1
        for k in range(-1, K):
            mask = labels == k
            plt.scatter(U[mask, 0], U[mask, 1], s=10, alpha=0.7,
                        label=("noise" if k == -1 else f"cluster {k}"))
        plt.legend(loc="best", fontsize=8)
        plt.title("UMAP of Δ (for intuition only)")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()

        # ALSO persist the embedding so other steps can reuse it
        np.save(out_png.with_suffix(".npy"), U)

        return U

    # ---------------- Statistical Significance ----------------
    def _statistical_significance(self, observed_top_persistence: float, null_results: Dict[str, Any]) -> Dict[str, Any]:
        """One-sided p-value: how often do nulls produce >= observed persistence? Also report Cohen's d and 95% CI from bootstrap."""
        shuffled_persistences = [r.get("top_H1_persistence", 0.0) for r in null_results.get("shuffled_pairing", [])]
        self_persistences = [
            null_results.get("self_diff", {}).get("H_minus_H_top", 0.0),
            null_results.get("self_diff", {}).get("T_minus_T_top", 0.0),
        ]
        all_nulls = [float(x) for x in (shuffled_persistences + self_persistences) if np.isfinite(x)]

        if not all_nulls:
            return {"p_value": 1.0, "significance": "insufficient_nulls"}

        # p-value (conservative, includes self-diffs)
        p_value = float(sum(1 for v in all_nulls if v >= observed_top_persistence) / len(all_nulls))

        # Effect size (Cohen's d vs nulls)
        null_mean = float(np.mean(all_nulls))
        null_std  = float(np.std(all_nulls) + 1e-8)
        cohen_d   = float((observed_top_persistence - null_mean) / null_std)

        # 95% CI from bootstrap distribution if available
        bs = [r.get("top_H1_persistence", 0.0) for r in self.stability_results.get("bootstrap", [])]
        if bs:
            ci_lower = float(np.percentile(bs, 2.5))
            ci_upper = float(np.percentile(bs, 97.5))
        else:
            ci_lower = ci_upper = float(observed_top_persistence)

        return {
            "p_value": p_value,
            "effect_size_cohens_d": cohen_d,
            "null_mean": null_mean,
            "null_std": null_std,
            "bootstrap_ci_95": [ci_lower, ci_upper],
            "significance_level": "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low" if p_value < 0.1 else "not_significant",
        }

    # ---------------- Multiple Testing (BH-lite note) ----------------
    def _correct_multiple_testing(self, h1_bars: list[list[float]], null_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pragmatic control: compute p-values for each H1 bar using the *same* null persistence
        distribution (of top persistence). This is conservative; we report BH-adjusted q-values.
        """
        if not h1_bars:
            return {"total_bars": 0, "significant_bars_bh_5pct": 0, "q_values": []}

        pers = np.array([float(d - b) for b, d in h1_bars], dtype=float)
        shuffled = [r.get("top_H1_persistence", 0.0) for r in null_results.get("shuffled_pairing", [])]
        selfs = [
            null_results.get("self_diff", {}).get("H_minus_H_top", 0.0),
            null_results.get("self_diff", {}).get("T_minus_T_top", 0.0),
        ]
        nulls = np.array([float(x) for x in (shuffled + selfs) if np.isfinite(x)], dtype=float)
        if nulls.size == 0:
            # fallback: no nulls -> cannot correct
            return {"total_bars": len(h1_bars), "significant_bars_bh_5pct": None, "q_values": []}

        # per-bar p-values (one-sided)
        pvals = np.array([(nulls >= p).mean() for p in pers], dtype=float)

        # Benjamini–Hochberg (BH) FDR control
        m = len(pvals)
        order = np.argsort(pvals)
        ranks = np.empty_like(order); ranks[order] = np.arange(1, m+1)
        qvals = pvals * m / ranks
        # monotone
        qvals_sorted = np.minimum.accumulate(qvals[order][::-1])[::-1]
        qvals_bh = np.empty_like(qvals); qvals_bh[order] = qvals_sorted

        sig_5 = int((qvals_bh <= 0.05).sum())
        sig_10 = int((qvals_bh <= 0.10).sum())

        return {
            "total_bars": int(m),
            "significant_bars_bh_5pct": sig_5,
            "significant_bars_bh_10pct": sig_10,
            "top_bar_persistence": float(pers.max() if m else 0.0),
            "q_values": [float(x) for x in qvals_bh],
        }

    # ---------------- Parameter Sensitivity ----------------
    def _parameter_sensitivity_analysis(self, Delta: np.ndarray, base_persistence: float) -> Dict[str, Any]:
        """Report how the top H1 persistence reacts to key knobs."""
        from ripser import ripser
        out: Dict[str, Any] = {"baseline_top_H1_persistence": float(base_persistence)}

        # maxdim sensitivity
        for md in (1, 2):
            try:
                dgms = ripser(Delta, maxdim=md)["dgms"]
                H1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
                top = 0.0 if H1.shape[0] == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
            except Exception:
                top = None
            out[f"maxdim_{md}"] = top

        # Weight jitter is already covered by _stability_checks; echo summary deltas
        if "weight_jitter" in self.stability_results:
            wj = [r.get("top_H1_persistence", 0.0) for r in self.stability_results["weight_jitter"]]
            if wj:
                out["weight_jitter_mean"] = float(np.mean(wj))
                out["weight_jitter_std"] = float(np.std(wj))
        return out

    # ---------------- Assumption/Quality Checks ----------------
    def _validate_tda_assumptions(self, Delta: np.ndarray) -> Dict[str, Any]:
        n_samples, n_features = int(Delta.shape[0]), int(Delta.shape[1])
        min_samples_for_tda = max(20, n_features * 2)

        # duplicates
        try:
            unique_points = len(np.unique(Delta, axis=0))
            duplicate_ratio = float((n_samples - unique_points) / max(1, n_samples))
        except Exception:
            duplicate_ratio = 0.0

        # IQR outlier flagging
        try:
            q75, q25 = np.percentile(Delta, [75, 25], axis=0)
            iqr = q75 - q25 + 1e-12
            outlier_mask = (Delta < (q25 - 1.5 * iqr)) | (Delta > (q75 + 1.5 * iqr))
            outlier_ratio = float(np.mean(outlier_mask))
        except Exception:
            outlier_ratio = 0.0

        # numeric conditioning (rough)
        try:
            cond_num = float(np.linalg.cond(Delta.T @ Delta + 1e-8 * np.eye(n_features)))
        except Exception:
            cond_num = float("inf")

        return {
            "sample_size_adequate": bool(n_samples >= min_samples_for_tda),
            "n_samples": n_samples,
            "n_features": n_features,
            "min_samples_needed": int(min_samples_for_tda),
            "duplicate_ratio": duplicate_ratio,
            "outlier_ratio": outlier_ratio,
            "condition_number": cond_num,
            "numerical_stability": "good" if np.isfinite(cond_num) and cond_num < 1e10 else "poor",
        }
