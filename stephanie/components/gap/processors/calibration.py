# stephanie/components/gap/processors/calibration.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from stephanie.components.gap.models import GapConfig

logger = logging.getLogger(__name__)


class CalibrationProcessor:
    """
    Fits a monotone piecewise-linear calibration Tiny->HRM per dimension and
    simulates a routing policy using Tiny diagnostics (uncertainty / OOD).
    Persists:
      - metrics/calibration_params.json
      - metrics/routing_detail.json
      - metrics/routing_summary.json
    """

    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger

    # ---------- Public API ----------------------------------------------------
    async def execute_calibration(
        self,
        analysis_results: Dict[str, Any],   # currently not required; kept for API symmetry
        run_id: str,
        progress_cb=None,
        alias_a: str = "HRM",
        alias_b: str = "Tiny",
    ) -> Dict[str, Any]:
        """
        Load aligned matrices (saved in scoring), fit per-dimension monotone calibration,
        build a Tiny-diagnostics routing mask, and compute MAE deltas vs HRM.

        Returns:
            {
              "params_path": "<.../calibration_params.json>",
              "routing_summary_path": "<.../routing_summary.json>",
              "routing_detail_path": "<.../routing_detail.json>",
              "usage_rate": <float>,
              "avg_mae_vs_hrm": <float>
            }
        """
        storage = self.container.get("storage")
        base_dir = Path(storage.base_dir) / run_id
        aligned_dir = base_dir / "aligned"
        metrics_dir = base_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # 0) Progress
        if progress_cb:
            progress_cb(0, {"stage": "calibration"})

        # 1) Load aligned matrices + names (as produced by ScoringProcessor)
        hrm_mat = np.load(aligned_dir / f"{alias_a}_matrix.npy")
        tiny_mat = np.load(aligned_dir / f"{alias_b}_matrix.npy")

        with open(aligned_dir / f"{alias_a}_metric_names.json", "r", encoding="utf-8") as f:
            hrm_names: List[str] = json.load(f)
        with open(aligned_dir / f"{alias_b}_metric_names.json", "r", encoding="utf-8") as f:
            tiny_names: List[str] = json.load(f)

        dims = list(self.config.dimensions)

        # 2) Per-dimension monotone PL calibration Tiny->HRM
        calib_params: Dict[str, Dict[str, List[float]]] = {}
        dim_stats: List[Dict[str, Any]] = []

        for dim in dims:
            i_hrm = _col_for_dim(hrm_names, "hrm", dim)
            i_tny = _col_for_dim(tiny_names, "tiny", dim)
            if i_hrm is None or i_tny is None:
                dim_stats.append({
                    "dimension": dim,
                    "status": "missing",
                    "hrm_col": i_hrm,
                    "tiny_col": i_tny,
                })
                continue

            hrm_scores = _safe01(hrm_mat[:, i_hrm])
            tiny_scores = _safe01(tiny_mat[:, i_tny])

            calib = _monotone_pl_calibration(tiny_scores, hrm_scores, n_knots=21)
            calib_params[dim] = calib

            mae_pre = _mae(tiny_scores, hrm_scores)
            tiny_cal = _apply_monotone_pl(tiny_scores, calib)
            mae_post = _mae(tiny_cal, hrm_scores)

            dim_stats.append({
                "dimension": dim,
                "status": "ok",
                "mae_pre": round(mae_pre, 6),
                "mae_post": round(mae_post, 6),
                "improved": bool(mae_post < mae_pre - 1e-6),
                "hrm_col": int(i_hrm),
                "tiny_col": int(i_tny),
            })

        # Persist calibration params
        params_path = metrics_dir / "calibration_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump({"per_dimension": calib_params, "stats": dim_stats}, f, indent=2)

        # 3) Build routing mask from Tiny diagnostics (uncertainty / ood_hat / consistency)
        #    We look for diagnostic columns on the Tiny side using loose search.
        tiny_unc = _maybe_diag_col(tiny_mat, tiny_names, ("uncertainty",))
        tiny_ood = _maybe_diag_col(tiny_mat, tiny_names, ("ood_hat",))
        tiny_unc = _safe01(tiny_unc) if tiny_unc is not None else np.zeros(tiny_mat.shape[0])
        tiny_ood = _safe01(tiny_ood) if tiny_ood is not None else np.zeros(tiny_mat.shape[0])

        thr_unc = float(getattr(self.config, "route_threshold_uncertainty", 0.6))
        thr_ood = float(getattr(self.config, "route_threshold_ood", 0.7))

        use_hrm_mask = (tiny_unc > thr_unc) | (tiny_ood > thr_ood)
        usage_rate = float(np.mean(use_hrm_mask))

        # 4) Simulate routed final scores and compute dimension MAEs vs HRM
        per_dim_results: List[Dict[str, Any]] = []
        for dim in dims:
            if dim not in calib_params:
                per_dim_results.append({"dimension": dim, "status": "skipped"})
                continue

            i_hrm = _col_for_dim(hrm_names, "hrm", dim)
            i_tny = _col_for_dim(tiny_names, "tiny", dim)
            if i_hrm is None or i_tny is None:
                per_dim_results.append({"dimension": dim, "status": "missing"})
                continue

            hrm_scores = _safe01(hrm_mat[:, i_hrm])
            tiny_scores = _safe01(tiny_mat[:, i_tny])
            tiny_cal = _apply_monotone_pl(tiny_scores, calib_params[dim])

            final = np.where(use_hrm_mask, hrm_scores, tiny_cal)

            mae_vs_hrm = _mae(final, hrm_scores)
            mae_tiny   = _mae(tiny_scores, hrm_scores)
            mae_cal    = _mae(tiny_cal, hrm_scores)

            per_dim_results.append({
                "dimension": dim,
                "status": "ok",
                "mae_vs_hrm_routed": round(mae_vs_hrm, 6),
                "mae_vs_hrm_tiny": round(mae_tiny, 6),
                "mae_vs_hrm_calibrated_tiny": round(mae_cal, 6),
            })

        # 5) Persist routing results
        detail_path = metrics_dir / "routing_detail.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump({"per_dimension": per_dim_results}, f, indent=2)

        maes = [r.get("mae_vs_hrm_routed") for r in per_dim_results if r.get("status") == "ok"]
        avg_mae = float(np.mean(maes)) if maes else 0.0

        summary_path = metrics_dir / "routing_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "usage_rate": round(usage_rate, 6),
                "avg_mae_vs_hrm": round(avg_mae, 6),
                "thresholds": {"uncertainty": thr_unc, "ood": thr_ood},
            }, f, indent=2)

        # 6) Final progress + return
        if progress_cb:
            progress_cb(100, 100, {"stage": "calibration"})

        return {
            "params_path": str(params_path),
            "routing_summary_path": str(summary_path),
            "routing_detail_path": str(detail_path),
            "usage_rate": usage_rate,
            "avg_mae_vs_hrm": avg_mae,
        }


# ---------- helpers (mirrors of the agent version) ---------------------------
def _find_first(names: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in names:
            return c
    return None

def _col_for_dim(names: List[str], model: str, dim: str) -> Optional[int]:
    """
    Find a column index for {model}.{dim} final score-like series.
    Preference: .score -> .aggregate -> exact -> any suffix match.
    """
    priorities = [
        f"{model}.{dim}.score",
        f"{model}.{dim}.aggregate",
        f"{model}.{dim}",
    ]
    choice = _find_first(names, priorities)
    if choice is None:
        # loose search fallback
        for i, n in enumerate(names):
            if n.startswith(f"{model}.{dim}") and n.endswith((".score", ".aggregate", ".value", ".raw")):
                return i
        return None
    return names.index(choice)

def _col_for_diag(names: List[str], model: str, key_parts: tuple[str, ...]) -> Optional[int]:
    """
    Find diagnostic column like 'tiny.reasoning.attr.ood_hat' or 'tiny.uncertainty'.
    Matches if all key_parts appear in the column name (in order).
    """
    def has_parts(s: str) -> bool:
        pos = 0
        for kp in key_parts:
            j = s.find(kp, pos)
            if j < 0:
                return False
            pos = j + len(kp)
        return True

    for i, n in enumerate(names):
        if n.startswith(f"{model}.") and has_parts(n):
            return i
    return None

def _maybe_diag_col(mat: np.ndarray, names: List[str], parts: tuple[str, ...]) -> Optional[np.ndarray]:
    idx = _col_for_diag(names, "tiny", parts)
    if idx is None:
        return None
    return mat[:, idx]

def _safe01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    return np.clip(a, 0.0, 1.0)

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a - b)))

def _monotone_pl_calibration(x: np.ndarray, y: np.ndarray, *, n_knots: int = 21) -> Dict[str, List[float]]:
    """
    Fit simple monotone piecewise-linear map Tiny->HRM using quantile knots.
    Returns {"x_knots":[...],"y_knots":[...]} with y non-decreasing in x.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return {"x_knots": [0.0, 1.0], "y_knots": [0.0, 1.0]}

    qs = np.linspace(0, 1, n_knots)
    x_knots = np.quantile(x, qs).astype(float)
    # hard clamp ends to [0,1] to stabilize extrapolation
    x_knots[0], x_knots[-1] = 0.0, 1.0

    y_knots = []
    for q in qs:
        lo = np.quantile(x, max(0.0, q - 0.025))
        hi = np.quantile(x, min(1.0, q + 0.025))
        mask = (x >= lo) & (x <= hi)
        if mask.sum() < 8:
            # nearest-neighbor fallback
            idx = np.argsort(np.abs(x - np.quantile(x, q)))[:16]
            y_knots.append(float(np.mean(y[idx])))
        else:
            y_knots.append(float(np.mean(y[mask])))

    # isotonic projection
    yk = np.array(y_knots, dtype=np.float64)
    for i in range(1, len(yk)):
        if yk[i] < yk[i - 1]:
            yk[i] = yk[i - 1]
    yk = np.clip(yk, 0.0, 1.0)

    return {"x_knots": x_knots.tolist(), "y_knots": yk.tolist()}

def _apply_monotone_pl(x: np.ndarray, calib: Dict[str, List[float]]) -> np.ndarray:
    if not calib or "x_knots" not in calib or "y_knots" not in calib:
        return x
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    xk = np.asarray(calib["x_knots"], dtype=np.float64)
    yk = np.asarray(calib["y_knots"], dtype=np.float64)
    return np.interp(xx, xk, yk).astype(np.float64)
