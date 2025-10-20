# stephanie/components/gap/processors/topology.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List, Literal

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors

from stephanie.utils.json_sanitize import dumps_safe
from stephanie.utils.progress_mixin import ProgressMixin

logger = logging.getLogger(__name__)
if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")


# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _zscore(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    return (M - mu) / (sd + eps)


# ---------------------------------------------------------------------
# Hole visualization (UMAP + representative loop)
# ---------------------------------------------------------------------
def plot_topology_holes(
    run_dir: Path,
    umap_xy: np.ndarray,
    X_delta: np.ndarray,
    H1_bars: List[Tuple[float, float]],
    max_edges: int = 200_000,
) -> Dict[str, Any]:
    """
    Visualize holes:
      1) UMAP scatter (density)
      2) Loop overlay using a representative cycle at eps ~ mid of top H1 bar
    """
    out: Dict[str, Any] = {}
    visuals = run_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)

    # (A) UMAP scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(umap_xy[:, 0], umap_xy[:, 1], s=3, alpha=0.35, linewidths=0)
    ax.set_title("UMAP of Δ-space (space between models)")
    ax.set_xticks([]); ax.set_yticks([])
    umap_png = visuals / "umap_delta_scatter.png"
    fig.savefig(umap_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    out["umap_scatter"] = str(umap_png)

    if not H1_bars:
        out["note"] = "No H1 bars → no loop overlay."
        return out

    # (B) pick the most persistent bar and choose eps = mid(birth, death)
    H1_bars_np = np.asarray(H1_bars, dtype=float)
    pers = H1_bars_np[:, 1] - H1_bars_np[:, 0]
    top_idx = int(np.argmax(pers))
    b, d = H1_bars_np[top_idx].tolist()
    eps = 0.5 * (b + d)
    out["top_H1_bar"] = {"birth": float(b), "death": float(d), "persistence": float(d - b)}
    out["chosen_eps"] = float(eps)

    # (C) Build eps-graph in Δ-space using kNN prefilter
    N = X_delta.shape[0]
    k = min(30, max(10, int(np.sqrt(max(1, N)))))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X_delta)
    dists, nbrs = nn.kneighbors(X_delta, return_distance=True)

    edges: List[Tuple[int, int, float]] = []
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

    # (D) Find a component with cycles; extract one simple cycle
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    cycle_nodes = None
    for comp in comps:
        sub = G.subgraph(comp).copy()
        if sub.number_of_edges() >= sub.number_of_nodes():
            cb = nx.cycle_basis(sub)
            if cb:
                cyc = max(cb, key=len)  # choose a long one for visibility
                cycle_nodes = list(cyc)
                break

    if cycle_nodes is None:
        out["note"] = "No cycle found in eps-graph at chosen eps (try nearby eps)."
        return out

    # (E) Overlay loop on UMAP
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(umap_xy[:, 0], umap_xy[:, 1], s=2, alpha=0.2, linewidths=0)
    loop_xy = umap_xy[np.array(cycle_nodes)]
    ax.plot(loop_xy[:, 0], loop_xy[:, 1], lw=2)
    ax.plot([loop_xy[-1, 0], loop_xy[0, 0]], [loop_xy[-1, 1], loop_xy[0, 1]], lw=2)
    ax.set_title(f"Topological Loop Overlay (ε={eps:.3f})")
    ax.set_xticks([]); ax.set_yticks([])
    loop_png = visuals / "umap_delta_loop_overlay.png"
    fig.savefig(loop_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    out["umap_loop_overlay"] = str(loop_png)
    out["cycle_nodes"] = [int(x) for x in cycle_nodes]

    # (F) Optional: highlight component nodes
    comp_nodes = list(comps[0]) if comps else []
    comp_mask = np.zeros(N, dtype=bool); comp_mask[comp_nodes] = True
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(umap_xy[~comp_mask, 0], umap_xy[~comp_mask, 1], s=2, alpha=0.05, linewidths=0)
    ax.scatter(umap_xy[comp_mask, 0], umap_xy[comp_mask, 1], s=4, alpha=0.35, linewidths=0)
    ax.plot(loop_xy[:, 0], loop_xy[:, 1], lw=2)
    ax.plot([loop_xy[-1, 0], loop_xy[0, 0]], [loop_xy[-1, 1], loop_xy[0, 1]], lw=2)
    ax.set_title("Component + Representative Loop")
    ax.set_xticks([]); ax.set_yticks([])
    comp_png = visuals / "umap_delta_component_and_loop.png"
    fig.savefig(comp_png, dpi=160, bbox_inches="tight"); plt.close(fig)
    out["umap_component_and_loop"] = str(comp_png)

    return out


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class TopologyConfig:
    # Which features to use – default to SCM core-5
    use_weighted: bool = True
    weights: Optional[Dict[str, float]] = None  # keys like "reasoning.score01"
    # UMAP/DBSCAN (storytelling only)
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
    # Repro
    random_seed: int = 42
    fast_mode: bool = False
    max_points_for_ph: Optional[int] = 1500
    compute_significance: bool = True
    # TDA backend: "ripser" (default) or "giotto"
    tda_backend: Literal["ripser", "giotto"] = "ripser"


# ---------------------------------------------------------------------
# PH backends (Ripser vs Giotto) — uniform API for H0/H1 bars
# ---------------------------------------------------------------------
class _PHBackend:
    def h0_h1(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class _RipserPH(_PHBackend):
    def h0_h1(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from ripser import ripser
        dgms = ripser(X, maxdim=1)["dgms"]
        H0 = dgms[0] if len(dgms) > 0 else np.zeros((0, 2), dtype=float)
        H1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2), dtype=float)
        return H0, H1


class _GiottoPH(_PHBackend):
    def __init__(self):
        from gtda.homology import VietorisRipsPersistence
        self.vr = VietorisRipsPersistence(homology_dimensions=[0, 1], metric="euclidean")

    def h0_h1(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D = X[np.newaxis, :, :]
        diags = self.vr.fit_transform(D)   # (1, n_points, 3)
        diag = diags[0] if diags.ndim == 3 else diags
        if diag.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)
        H0 = diag[diag[:, 2] == 0][:, :2]
        H1 = diag[diag[:, 2] == 1][:, :2]
        return H0.astype(float), H1.astype(float)


# ---------------------------------------------------------------------
# Processor (ProgressMixin version, no callbacks)
# ---------------------------------------------------------------------
class TopologyProcessor(ProgressMixin):
    """
    Computes persistent homology on the gap field Δ = H - T (SCM core-5),
    with stability + null controls, and generates UMAP storytelling figures.

    Outputs:
      visuals/pers_diagram_H1.png
      visuals/pers_barcode_H1.png
      visuals/umap_loop_overlay.png
      visuals/umap_loop_overlay.npy
      aligned/delta_core5.npy
      metrics/betti.json
      metrics/stability.json
      metrics/nulls.json
      metrics/tda_assumptions.json
      metrics/statistical_significance.json
      metrics/multiple_testing_correction.json
      metrics/parameter_sensitivity.json
    """

    def __init__(self, cfg: TopologyConfig, container, logger):
        self.cfg = cfg
        self.container = container
        self.logger = logger
        self.stability_results: Dict[str, Any] = {}

        # Init progress service (no callbacks passed around)
        self._init_progress(container, logger)

        # pick PH backend (configurable; default ripser)
        if getattr(self.cfg, "tda_backend", "ripser") == "giotto":
            try:
                self._ph_backend = _GiottoPH()
                self.logger.log("TopologyPHBackend", {"backend": "giotto-tda"})
            except Exception as e:
                self.logger.log("TopologyPHBackendFallback", {"backend": "ripser", "error": str(e)})
                self._ph_backend = _RipserPH()
        else:
            self._ph_backend = _RipserPH()

        # Fast mode softeners
        if self.cfg.fast_mode:
            self.cfg.n_bootstrap = min(self.cfg.n_bootstrap, 5)
            self.cfg.n_nulls = min(self.cfg.n_nulls, 20)
            self.cfg.umap_n_neighbors = min(self.cfg.umap_n_neighbors, 10)
            self.cfg.dbscan_min_samples = max(3, self.cfg.dbscan_min_samples // 2)

    async def run(self, run_id: str, base_dir: Path) -> Dict[str, Any]:
        task = f"topology:{run_id}"
        # High-level stages we’ll tick through:
        total_stages = 8  # load, delta, assumptions, ph, stability, nulls, umap, significance
        self.pstart(task, total=total_stages, meta={"run_id": run_id})

        # RNG control for reproducibility
        rng_state = np.random.get_state()
        np.random.seed(self.cfg.random_seed)
        try:
            base_dir = Path(base_dir)
            vis_dir = base_dir / run_id / "visuals"
            met_dir = base_dir / run_id / "metrics"
            aligned_dir = base_dir / run_id / "aligned"
            vis_dir.mkdir(parents=True, exist_ok=True)
            met_dir.mkdir(parents=True, exist_ok=True)

            # --- Load SCM-aligned matrices
            self.pstage(task, "load:start")
            H_path = aligned_dir / "hrm_scm_matrix.npy"
            T_path = aligned_dir / "tiny_scm_matrix.npy"
            names_path = aligned_dir / "scm_metric_names.json"

            if not H_path.exists() or not T_path.exists():
                self.logger.log("TopologyMissingMatrices", {
                    "run_id": run_id, "hrm_scm_matrix": str(H_path), "tiny_scm_matrix": str(T_path)
                })
                self.pstage(task, "load:missing", run_id=run_id)
                self.pdone(task, status="error")
                return {"status": "missing_scm_matrices"}

            H = np.load(H_path); T = np.load(T_path)
            if names_path.exists():
                with open(names_path, "r", encoding="utf-8") as f:
                    names = json.load(f)
            else:
                names = [
                    "scm.reasoning.score01","scm.knowledge.score01","scm.clarity.score01",
                    "scm.faithfulness.score01","scm.coverage.score01",
                    "scm.aggregate01","scm.uncertainty01","scm.ood_hat01","scm.consistency01",
                    "scm.length_norm01","scm.temp01","scm.agree_hat01"
                ]
            self.pstage(task, "load:done", rows=int(H.shape[0]), cols=int(H.shape[1]))
            self.ptick(task, done=1, total=total_stages)

            # --- Δ cloud build (core-5, zscore, weights)
            self.pstage(task, "delta:prep")
            want = [
                "scm.reasoning.score01","scm.knowledge.score01","scm.clarity.score01",
                "scm.faithfulness.score01","scm.coverage.score01"
            ]
            name_to_idx = {n: i for i, n in enumerate(names)}
            col_idx = [name_to_idx[w] for w in want if w in name_to_idx]
            if len(col_idx) != 5:
                col_idx = list(range(min(5, H.shape[1])))

            H5 = H[:, col_idx]; T5 = T[:, col_idx]
            if H5.shape != T5.shape or H5.shape[1] == 0:
                self.logger.log("TopologyShapeMismatch", {"H5": H5.shape, "T5": T5.shape})
                self.pstage(task, "delta:shape_mismatch")
                self.pdone(task, status="error")
                return {"status": "shape_mismatch"}

            if self.cfg.zscore_inputs:
                H5 = _zscore(H5); T5 = _zscore(T5)
            Delta = H5 - T5
            # Optional per-dimension weights
            if self.cfg.use_weighted and self.cfg.weights:
                W = np.ones(Delta.shape[1], dtype=np.float32)
                for i, n in enumerate([names[j] for j in col_idx]):
                    suf = n.split("scm.")[-1]
                    W[i] = float(self.cfg.weights.get(suf, 1.0))
                Delta = Delta * W[None, :]
            np.save(aligned_dir / "delta_core5.npy", Delta)
            self.pstage(task, "delta:ready", shape=list(Delta.shape))
            self.ptick(task, done=2, total=total_stages)

            # Optional cap for heavy steps
            row_cap = self.cfg.max_points_for_ph or 0
            if row_cap and Delta.shape[0] > row_cap:
                idx = np.random.RandomState(self.cfg.random_seed).choice(Delta.shape[0], size=row_cap, replace=False)
                Delta_ph = Delta[idx]
            else:
                idx = None
                Delta_ph = Delta

            # --- Assumption/quality checks
            self.pstage(task, "assumptions:start")
            tda_assumptions = self._validate_tda_assumptions(Delta)
            _save_json(met_dir / "tda_assumptions.json", tda_assumptions)
            self.pstage(task, "assumptions:done", numerical_stability=tda_assumptions.get("numerical_stability"))
            self.ptick(task, done=3, total=total_stages)

            # --- PH + figures
            self.pstage(task, "ph:start")
            betti = self._compute_ph_and_figures(Delta_ph, vis_dir)
            _save_json(met_dir / "betti.json", betti)
            self.pstage(task, "ph:done", b1=betti.get("b1", 0))
            self.ptick(task, done=4, total=total_stages)

            # --- Stability (bootstrap + weight jitter)
            self.pstage(task, "stability:start")
            stability = self._stability_checks(Delta_ph, names=[names[j] for j in col_idx], vis_dir=vis_dir)
            self.stability_results = stability
            _save_json(met_dir / "stability.json", stability)
            self.pstage(task, "stability:done")
            self.ptick(task, done=5, total=total_stages)

            # --- Null controls (simple + stronger)
            self.pstage(task, "nulls:start")
            null_simple = self._null_controls(H5, T5)
            self.pstage(task, "nulls:simple:done")
            null_strong = self._null_controls_stronger(Delta_ph)
            self.pstage(task, "nulls:strong:done")
            nulls = {**null_simple, **null_strong}
            _save_json(met_dir / "nulls.json", nulls)
            self.ptick(task, done=6, total=total_stages)

            # --- UMAP storytelling + loop overlay
            self.pstage(task, "umap:start")
            U = None
            loop_vis = {}
            try:
                U = self._umap_overlay(Delta, out_png=vis_dir / "umap_loop_overlay.png")
            except Exception as e:
                logger.info(f"[Topology] UMAP overlay skipped: {e}")

            try:
                if U is not None and betti.get("H1_bars"):
                    loop_vis = plot_topology_holes(
                        run_dir=base_dir / run_id,
                        umap_xy=U,
                        X_delta=Delta,
                        H1_bars=[tuple(b) for b in betti["H1_bars"]],
                        max_edges=200_000,
                    )
                    self.logger.log("TopologyLoopOverlay", loop_vis)

                    cycle_nodes = [int(i) for i in loop_vis.get("cycle_nodes", [])]
                    # Per-dimension semantics on the loop (SCM core-5)
                    loop_sem = self._loop_semantics_report(
                        H5=H5, T5=T5,
                        cycle_nodes=cycle_nodes,
                        dim_names=["reasoning","knowledge","clarity","faithfulness","coverage"],
                        base_dir=base_dir / run_id,
                        k_examples=50,
                    )
                    _save_json(met_dir / "loop_semantics.json", loop_sem)

                    if cycle_nodes:
                        score = self._loop_circularity_score(Delta[cycle_nodes])
                        _save_json(met_dir / "loop_shape.json", {"circularity_score": score})

                    cases = self._export_loop_cases(
                        run_dir=base_dir / run_id,
                        cycle_nodes=cycle_nodes,
                        H5=H5, T5=T5,
                        dim_names=["reasoning","knowledge","clarity","faithfulness","coverage"],
                        k_examples=50,
                    )
                    self.logger.log("TopologyLoopCases", cases)
            except Exception as e:
                logger.info(f"[Topology] Loop overlay skipped: {e}")
            self.pstage(task, "umap:done", has_loop=bool(betti.get("H1_bars")))
            self.ptick(task, done=7, total=total_stages)

            # --- Formal significance & corrections
            self.pstage(task, "significance:start")
            if self.cfg.compute_significance and betti.get("b1", 0) > 0:
                significance = self._statistical_significance(betti["top_H1_persistence"], nulls)
                _save_json(met_dir / "statistical_significance.json", significance)

                mult_test = self._correct_multiple_testing(betti.get("H1_bars", []), nulls)
                _save_json(met_dir / "multiple_testing_correction.json", mult_test)

                param_sens = self._parameter_sensitivity_analysis(Delta_ph, betti["top_H1_persistence"])
                _save_json(met_dir / "parameter_sensitivity.json", param_sens)
            else:
                significance = {"p_value": 1.0, "significance": "no_h1_bars"}
                mult_test = {"total_bars": 0, "significant_bars_bh_5pct": 0, "q_values": []}
                param_sens = {}
            self.pstage(task, "significance:done", b1=betti.get("b1", 0))
            self.ptick(task, done=8, total=total_stages)

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
            self.pdone(task, status="ok")
            return out

        finally:
            np.random.set_state(rng_state)

    # ---------------------------------------------------------------------
    # Core PH
    # ---------------------------------------------------------------------
    def _compute_ph_and_figures(self, Delta: np.ndarray, vis_dir: Path) -> Dict[str, Any]:
        # Compute diagrams via selected backend
        H0, H1 = self._ph_backend.h0_h1(Delta)

        # All diagrams (optional)
        try:
            plt.figure(figsize=(6, 5))
            if H0.size:
                plt.scatter(H0[:, 0], H0[:, 1], s=8, alpha=0.6, label="H0")
            if H1.size:
                plt.scatter(H1[:, 0], H1[:, 1], s=8, alpha=0.8, label="H1")
            lim = np.nanmax([H0[:, :2].max() if H0.size else 0, H1[:, :2].max() if H1.size else 0])
            plt.plot([0, lim], [0, lim], ls="--", lw=1, alpha=0.5, color="k")
            plt.legend()
            plt.title("Persistence Diagrams (H0/H1)")
            vis_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(vis_dir / "pers_diagram_all.png", dpi=160, bbox_inches="tight"); plt.close()
        except Exception:
            pass

        # H1-only diagram and barcode
        try:
            plt.figure(figsize=(6, 5))
            if H1.size:
                plt.scatter(H1[:, 0], H1[:, 1], s=10, alpha=0.9)
                lim = float(max(H1.max(), 1e-6))
                plt.plot([0, lim], [0, lim], ls="--", lw=1, alpha=0.5, color="k")
            plt.title("Persistence Diagram (H1)")
            plt.savefig(vis_dir / "pers_diagram_H1.png", dpi=160, bbox_inches="tight"); plt.close()

            if H1.size:
                lives = H1[:, 1] - H1[:, 0]
                order = np.argsort(lives)[::-1]
                H1_sorted = H1[order]
                plt.figure(figsize=(8, max(3, 0.25 * len(H1_sorted))))
                for i, (b, d) in enumerate(H1_sorted):
                    plt.hlines(y=i, xmin=b, xmax=d, linewidth=2)
                plt.xlabel("Filtration scale"); plt.ylabel("H1 features (sorted)")
                plt.title("Persistence Barcode (H1)")
                plt.savefig(vis_dir / "pers_barcode_H1.png", dpi=160, bbox_inches="tight"); plt.close()
        except Exception:
            pass

        b0 = int(H0.shape[0])
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
    def _stability_checks(self, Delta: np.ndarray, names: List[str], vis_dir: Path) -> Dict[str, Any]:
        N = Delta.shape[0]
        bstrap: List[Dict[str, Any]] = []

        B = max(1, int(self.cfg.n_bootstrap))
        for b in range(B):
            m = max(2, int(self.cfg.bootstrap_frac * N))
            idx = np.random.choice(N, size=m, replace=False)
            dsub = Delta[idx]
            _, H1 = self._ph_backend.h0_h1(dsub)
            top = 0.0 if H1.shape[0] == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
            bstrap.append({"b1": int(H1.shape[0]), "top_H1_persistence": top})
            if (b % 2) == 0 or (b + 1) == B:
                self.pstage(f"topology:bootstrap", "tick", b=b+1, B=B)

        jitter: List[Dict[str, Any]] = []
        if self.cfg.use_weighted:
            J = 10
            for j in range(J):
                r = 1.0 + 0.1 * (2 * np.random.rand(Delta.shape[1]) - 1)
                dJ = Delta * r[None, :]
                _, H1 = self._ph_backend.h0_h1(dJ)
                top = 0.0 if H1.shape[0] == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
                jitter.append({"b1": int(H1.shape[0]), "top_H1_persistence": top})
                if (j % 2) == 0 or (j + 1) == J:
                    self.pstage(f"topology:jitter", "tick", j=j+1, J=J)

        return {"bootstrap": bstrap, "weight_jitter": jitter}

    # ---------------------------------------------------------------------
    # Export loop cases
    # ---------------------------------------------------------------------
    def _export_loop_cases(
        self,
        run_dir: Path,
        cycle_nodes: list[int],
        H5: np.ndarray,
        T5: np.ndarray,
        dim_names: list[str],
        k_examples: int = 50,
    ) -> dict:
        out_dir = run_dir / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)

        prov_path = run_dir / "raw" / "row_provenance.json"
        if not prov_path.exists():
            return {"status": "no_provenance"}

        with open(prov_path, "r", encoding="utf-8") as f:
            provenance = json.load(f)

        idxs = [int(i) for i in cycle_nodes]
        if not idxs:
            return {"status": "no_cycle_nodes"}

        rows = []
        for i in idxs[:k_examples]:
            meta = provenance[i] if i < len(provenance) else {"row_index": i}
            rec = {
                "row_index": i,
                "node_id": meta.get("node_id"),
                "dimension": meta.get("dimension"),
                "goal_text": meta.get("goal_text"),
                "output_text": meta.get("output_text"),
            }
            for j, name in enumerate(dim_names):
                rec[f"hrm.{name}"]  = float(H5[i, j])
                rec[f"tiny.{name}"] = float(T5[i, j])
                rec[f"delta.{name}"] = float(H5[i, j] - T5[i, j])
            d = H5[i, :] - T5[i, :]
            rec["delta_l2_core5"] = float(np.linalg.norm(d, ord=2))
            rows.append(rec)

        import csv
        csv_path  = out_dir / "loop_cases.csv"
        json_path = out_dir / "loop_cases.json"

        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(dumps_safe(rows, indent=2))

        return {"status": "ok", "count": len(rows), "csv": str(csv_path), "json": str(json_path)}

    # ---------------------------------------------------------------------
    # Nulls (simple)
    # ---------------------------------------------------------------------
    def _null_controls(self, H5: np.ndarray, T5: np.ndarray) -> Dict[str, Any]:
        N = H5.shape[0]
        K = max(5, int(self.cfg.n_nulls))

        # Shuffled pairing H - T_perm
        shuffled: List[Dict[str, Any]] = []
        for k in range(K):
            perm = np.random.permutation(N)
            d = H5 - T5[perm]
            _, H1 = self._ph_backend.h0_h1(d)
            top = 0.0 if H1.shape[0] == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
            shuffled.append({"b1": int(H1.shape[0]), "top_H1_persistence": top})
            if (k % 5) == 0 or (k + 1) == K:
                self.pstage("topology:nulls:simple", "tick", k=k+1, K=K)

        # Self-differences (should be trivial)
        _, H1_selfH = self._ph_backend.h0_h1(H5 - H5)
        selfH_top = 0.0 if H1_selfH.shape[0] == 0 else float(np.max(H1_selfH[:, 1] - H1_selfH[:, 0]))
        _, H1_selfT = self._ph_backend.h0_h1(T5 - T5)
        selfT_top = 0.0 if H1_selfT.shape[0] == 0 else float(np.max(H1_selfT[:, 1] - H1_selfT[:, 0]))

        return {
            "shuffled_pairing": shuffled,
            "self_diff": {"H_minus_H_top": selfH_top, "T_minus_T_top": selfT_top},
        }

    # ---------------------------------------------------------------------
    # Nulls (stronger)
    # ---------------------------------------------------------------------
    def _null_controls_stronger(self, Delta: np.ndarray) -> Dict[str, Any]:
        from ripser import ripser  # used only for direct call in this function
        N = Delta.shape[0]
        K = max(50, int(self.cfg.n_nulls))
        res: Dict[str, Any] = {}

        # A) Sign-flip (Rademacher) nulls
        rads: List[Dict[str, Any]] = []
        for k in range(K):
            sigma = (np.random.rand(N) < 0.5).astype(np.float32) * 2 - 1  # ±1
            d = Delta * sigma[:, None]
            H1 = ripser(d, maxdim=self.cfg.max_betti_dim)["dgms"][1] if self.cfg.max_betti_dim >= 1 else np.zeros((0, 2))
            top = 0.0 if H1.shape[0] == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
            rads.append({"top_H1_persistence": top})
            if (k % 5) == 0 or (k + 1) == K:
                self.pstage("topology:nulls:signflip", "tick", k=k+1, K=K)
        res["sign_flip"] = rads

        # B) Gaussian surrogate with matched covariance
        mu = Delta.mean(axis=0, keepdims=True)
        Sigma = np.cov((Delta - mu).T)
        gauss: List[Dict[str, Any]] = []
        for k in range(K):
            g = np.random.multivariate_normal(mean=np.zeros(Delta.shape[1]), cov=Sigma, size=N)
            H1 = ripser(g, maxdim=self.cfg.max_betti_dim)["dgms"][1] if self.cfg.max_betti_dim >= 1 else np.zeros((0, 2))
            top = 0.0 if H1.shape[0] == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
            gauss.append({"top_H1_persistence": top})
        res["gaussian_cov"] = gauss

        return res

    # ---------------------------------------------------------------------
    # UMAP – storytelling (not used for homology)
    # ---------------------------------------------------------------------
    def _umap_overlay(self, Delta: np.ndarray, out_png: Path) -> np.ndarray:
        import umap
        from sklearn.cluster import DBSCAN

        reducer = umap.UMAP(
            n_neighbors=self.cfg.umap_n_neighbors,
            min_dist=self.cfg.umap_min_dist,
            metric="euclidean",
            n_jobs=1,           
            random_state=self.cfg.random_seed,
        )
        U = reducer.fit_transform(Delta)  # (N, 2)
        self.pstage("topology:umap", "fit:done", n=int(U.shape[0]))
        cl = DBSCAN(eps=self.cfg.dbscan_eps, min_samples=self.cfg.dbscan_min_samples)
        labels = cl.fit_predict(U)
        self.pstage("topology:umap", "cluster:done", clusters=int(max(labels) + 1 if labels.size else 0))

        plt.figure(figsize=(7, 6))
        K = max(labels) + 1 if labels.size else 0
        for k in range(-1, K):
            mask = labels == k
            plt.scatter(U[mask, 0], U[mask, 1], s=10, alpha=0.7,
                        label=("noise" if k == -1 else f"cluster {k}"))
        if K > 0:
            plt.legend(loc="best", fontsize=8)
        plt.title("UMAP of Δ (for intuition only)")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()

        np.save(out_png.with_suffix(".npy"), U)
        return U

    # ---------------------------------------------------------------------
    # Loop analytics
    # ---------------------------------------------------------------------
    def _loop_semantics_report(
        self,
        H5: np.ndarray,
        T5: np.ndarray,
        cycle_nodes: list[int],
        dim_names: list[str],
        base_dir: Path,
        *,
        k_examples: int = 8,
    ) -> dict:
        out_dir = base_dir / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not cycle_nodes:
            report = {"status": "no_cycle", "dims": dim_names, "examples": []}
            (out_dir / "loop_semantics.json").write_text(dumps_safe(report, indent=2))
            return report

        loop_H = H5[cycle_nodes, :]
        loop_T = T5[cycle_nodes, :]
        D = loop_H - loop_T  # Δ on the loop

        stats = []
        for j, name in enumerate(dim_names):
            col = D[:, j]
            stats.append({
                "dimension": name,
                "mean_delta": float(np.mean(col)),
                "abs_mean_delta": float(np.mean(np.abs(col))),
                "std_delta": float(np.std(col)),
                "min_delta": float(np.min(col)),
                "max_delta": float(np.max(col)),
            })

        key_dim = max(stats, key=lambda r: r["abs_mean_delta"])["dimension"]
        j_key = dim_names.index(key_dim)
        order = np.argsort(-np.abs(D[:, j_key]))
        take = min(k_examples, len(order))
        chosen = [int(cycle_nodes[i]) for i in order[:take]]

        report = {
            "status": "ok",
            "loop_size": int(len(cycle_nodes)),
            "dims": dim_names,
            "summary": stats,
            "most_divergent_dimension": key_dim,
            "example_row_indices": chosen,
        }

        (out_dir / "loop_semantics.json").write_text(dumps_safe(report, indent=2))
        return report

    # ---------------------------------------------------------------------
    # Statistical Significance
    # ---------------------------------------------------------------------
    def _statistical_significance(self, observed_top_persistence: float, null_results: Dict[str, Any]) -> Dict[str, Any]:
        shuffled = [r.get("top_H1_persistence", 0.0) for r in null_results.get("shuffled_pairing", [])]
        self_persistences = [
            null_results.get("self_diff", {}).get("H_minus_H_top", 0.0),
            null_results.get("self_diff", {}).get("T_minus_T_top", 0.0),
        ]
        signflip = [r.get("top_H1_persistence", 0.0) for r in null_results.get("sign_flip", [])]
        gauss = [r.get("top_H1_persistence", 0.0) for r in null_results.get("gaussian_cov", [])]

        all_nulls = [float(x) for x in (shuffled + self_persistences + signflip + gauss) if np.isfinite(x)]
        if not all_nulls:
            return {"p_value": 1.0, "significance": "insufficient_nulls"}

        p_value = float(sum(1 for v in all_nulls if v >= observed_top_persistence) / len(all_nulls))
        null_mean = float(np.mean(all_nulls))
        null_std = float(np.std(all_nulls) + 1e-8)
        cohen_d = float((observed_top_persistence - null_mean) / null_std)

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

    # ---------------------------------------------------------------------
    # Multiple Testing (BH FDR)
    # ---------------------------------------------------------------------
    def _correct_multiple_testing(self, h1_bars: List[List[float]], null_results: Dict[str, Any]) -> Dict[str, Any]:
        if not h1_bars:
            return {"total_bars": 0, "significant_bars_bh_5pct": 0, "q_values": []}

        pers = np.array([float(d - b) for b, d in h1_bars], dtype=float)

        shuffled = [r.get("top_H1_persistence", 0.0) for r in null_results.get("shuffled_pairing", [])]
        selfs = [
            null_results.get("self_diff", {}).get("H_minus_H_top", 0.0),
            null_results.get("self_diff", {}).get("T_minus_T_top", 0.0),
        ]
        signflip = [r.get("top_H1_persistence", 0.0) for r in null_results.get("sign_flip", [])]
        gauss = [r.get("top_H1_persistence", 0.0) for r in null_results.get("gaussian_cov", [])]
        nulls = np.array([float(x) for x in (shuffled + selfs + signflip + gauss) if np.isfinite(x)], dtype=float)
        if nulls.size == 0:
            return {"total_bars": len(h1_bars), "significant_bars_bh_5pct": None, "q_values": []}

        pvals = np.array([(nulls >= p).mean() for p in pers], dtype=float)
        m = len(pvals)
        order = np.argsort(pvals)
        ranks = np.empty_like(order); ranks[order] = np.arange(1, m + 1)
        qvals = pvals * m / ranks
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

    # ---------------------------------------------------------------------
    # Parameter Sensitivity
    # ---------------------------------------------------------------------
    def _parameter_sensitivity_analysis(self, Delta: np.ndarray, base_persistence: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {"baseline_top_H1_persistence": float(base_persistence)}

        for md in (1, 2):
            try:
                _, H1 = self._ph_backend.h0_h1(Delta)
                top = 0.0 if H1.shape[0] == 0 else float(np.max(H1[:, 1] - H1[:, 0]))
            except Exception:
                top = None
            out[f"maxdim_{md}"] = top

        if "weight_jitter" in self.stability_results:
            wj = [r.get("top_H1_persistence", 0.0) for r in self.stability_results["weight_jitter"]]
            if wj:
                out["weight_jitter_mean"] = float(np.mean(wj))
                out["weight_jitter_std"] = float(np.std(wj))
        return out

    # ---------------------------------------------------------------------
    # Assumption checks
    # ---------------------------------------------------------------------
    def _validate_tda_assumptions(self, Delta: np.ndarray) -> Dict[str, Any]:
        n_samples, n_features = int(Delta.shape[0]), int(Delta.shape[1])
        min_samples_for_tda = max(20, n_features * 2)

        # duplicates
        try:
            unique_points = len(np.unique(Delta, axis=0))
            duplicate_ratio = float((n_samples - unique_points) / max(1, n_samples))
        except Exception:
            duplicate_ratio = 0.0

        # IQR outliers
        try:
            q75, q25 = np.percentile(Delta, [75, 25], axis=0)
            iqr = q75 - q25 + 1e-12
            outlier_mask = (Delta < (q25 - 1.5 * iqr)) | (Delta > (q75 + 1.5 * iqr))
            outlier_ratio = float(np.mean(outlier_mask))
        except Exception:
            outlier_ratio = 0.0

        # numeric conditioning
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

    def _loop_circularity_score(self, Delta_loop: np.ndarray) -> float:
        # PCA to 2D
        X = Delta_loop - Delta_loop.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Y = X @ Vt[:2].T  # [m,2]
        ang = np.arctan2(Y[:,1], Y[:,0])
        re = np.mean(np.exp(1j * ang))
        circ_var = 1.0 - np.abs(re)   # 0 = perfect circle, 1 = spread
        return float(1.0 - circ_var)  # higher is “more circular”
