# stephanie/components/gap/processors/topology.py
from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        try:
            self._umap_overlay(Delta, out_png=vis_dir / "umap_loop_overlay.png")
        except Exception as e:
            logger.info(f"[Topology] UMAP overlay skipped: {e}")

        out = {
            "status": "ok",
            "betti": betti,
            "stability": stability,
            "nulls": nulls,
            "figures": {
                "H1_persistence_diagram": str((vis_dir / "pers_diagram_H1.png").resolve()),
                "H1_barcode": str((vis_dir / "pers_barcode_H1.png").resolve()),
                "umap_overlay": str((vis_dir / "umap_loop_overlay.png").resolve()),
            }
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
    def _umap_overlay(self, Delta: np.ndarray, out_png: Path) -> None:
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
            plt.scatter(U[mask, 0], U[mask, 1], s=10, alpha=0.7, label=("noise" if k == -1 else f"cluster {k}"))
        plt.legend(loc="best", fontsize=8)
        plt.title("UMAP of Δ (for intuition only)")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()
