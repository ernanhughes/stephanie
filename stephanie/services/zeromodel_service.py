# stephanie/services/zero_model_service.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spstats  # optional, but nice if available
from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.tools.gif_logger import GifLogger
from zeromodel.tools.spatial_optimizer import SpatialOptimizer

from stephanie.services.event_service import EventService
from stephanie.services.service_protocol import Service
from stephanie.utils.json_sanitize import dumps_safe
from stephanie.zeromodel.vpm_phos import robust01

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

log = logging.getLogger(__name__)

_DEFAULT_PIPELINE = [
    {"stage": "normalize", "params": {}},
    {"stage": "feature_engineering", "params": {}},
    {"stage": "organization", "params": {"strategy": "spatial"}},
]


# --------------------------------------------------------------------------- #
# INTERNAL SESSION STRUCTURE
# --------------------------------------------------------------------------- #
@dataclass
class _TimelineSession:
    run_id: str
    metrics_order: List[str]
    rows: List[List[float]] = field(default_factory=list)
    meta: List[Dict[str, Any]] = field(default_factory=list)
    out_dir: str = "data/vpms"

    def as_matrix(self) -> np.ndarray:
        """Return a well-formed float32 matrix, padding/truncating rows if necessary."""
        if not self.rows:
            return np.zeros((0, len(self.metrics_order)), dtype=np.float32)

        max_len = len(self.metrics_order)
        clean_rows = []
        for row in self.rows:
            if len(row) < max_len:
                row = list(row) + [0.0] * (max_len - len(row))
            elif len(row) > max_len:
                row = list(row)[:max_len]
            clean_rows.append(row)
        return np.asarray(clean_rows, dtype=np.float32)

def _rank_intensity(vec: np.ndarray, names: list[str] | None = None):
    """Return sorted indices by mean |intensity| (desc) plus a labeled list."""
    idx = np.argsort(-vec)  # desc
    if names:
        labeled = [{"metric": names[i], "mean_abs": float(vec[i]), "rank": r+1}
                   for r, i in enumerate(idx)]
    else:
        labeled = [{"metric": f"metric_{i}", "mean_abs": float(vec[i]), "rank": r+1}
                   for r, i in enumerate(idx)]
    return idx.tolist(), labeled

def _column_intensity(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.ndim < 2: M = M.reshape(1, -1)
    return np.mean(np.abs(M), axis=0)

def _row_intensity(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.ndim < 2: M = M.reshape(1, -1)
    return np.mean(np.abs(M), axis=1)

def _canonical_key(colname: str) -> str:
    """
    Map 'hrm.reasoning.score' -> 'reasoning', 'tiny.clarity' -> 'clarity'.
    Keeps only the segment after the first '.' and before the next '.' if present.
    """
    if not isinstance(colname, str): return ""
    parts = colname.split(".", 2)
    return parts[1] if len(parts) >= 2 else colname

def _center_and_whiten(X, eps=1e-6):
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Xw = U @ np.diag(1.0 / (S + eps))  # whitened rows
    return Xw, Vt, S

def _cca_shared_projections(A, B, k=20):
    # Whiten both views over the same rows
    A0, Va, Sa = _center_and_whiten(A)
    B0, Vb, Sb = _center_and_whiten(B)
    # Cross-covariance in whitened space
    C = A0.T @ B0
    U, S, Vt = np.linalg.svd(C, full_matrices=False)  # canonical directions
    k = int(min(k, U.shape[1], Vt.shape[0]))
    Wa = Va.T @ U[:, :k]         # map A->shared latent
    Wb = Vb.T @ Vt[:k, :].T      # map B->shared latent
    return Wa, Wb, S[:k]

def _normalize01(X):
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    if X.size == 0:
        return X
    mx = np.max(np.abs(X)) + 1e-8
    return X / mx

def _phos_pack_row(row):
    # sort-desc + square-pack (simple version)
    v = np.asarray(row, dtype=np.float32).ravel()
    v = v - np.percentile(v, 10)
    v = np.clip(v / (np.percentile(v, 90) - np.percentile(v, 10) + 1e-6), 0, 1)
    order = np.argsort(v)[::-1]
    v = v[order]
    s = int(np.ceil(np.sqrt(v.size)))
    pad = s*s - v.size
    if pad > 0: v = np.pad(v, (0, pad))
    return v.reshape(s, s)

def _phos_mean_image(M):
    # Average PHOS of rows to show model “energy” distribution
    imgs = [_phos_pack_row(r) for r in M]
    s = imgs[0].shape[0]
    imgs = [im if im.shape==(s,s) else np.zeros((s,s), dtype=np.float32) for im in imgs]
    return np.mean(np.stack(imgs, axis=0), axis=0)


# --- helper (put near other helpers in ZeroModelService) ---
def _pad_square_top_left(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape
    out = np.zeros((side, side), dtype=img.dtype)
    out[:h, :w] = img  # keep TL structure; pad BR
    return out

def _save_field_image(
    mat: np.ndarray,
    title: str,
    out_base: str,
    *,
    cmap: str = "coolwarm",
) -> Optional[str]:
    """
    Save a 2D heatmap image. Accepts 1D arrays (reshapes to Nx1).
    Returns the file path or None if input is empty.
    """
    mat = np.asarray(mat)
    if mat.size == 0:
        return None
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="Δ Intensity")
    plt.tight_layout()
    out_path = f"{out_base}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def _align_square_phos(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = max(A.shape[0], B.shape[0])
    if A.shape != (s, s): A = _pad_square_top_left(A, s)
    if B.shape != (s, s): B = _pad_square_top_left(B, s)
    return A, B

# --------------------------------------------------------------------------- #
# MAIN SERVICE
# --------------------------------------------------------------------------- #
class ZeroModelService(Service):
    """
    One service to rule them all:
    - Agent-controlled timelines: open → append → finalize
    - Optional bus-attached mode: attach/detach subjects per run
    - Rendering delegated to ZeroModel pipeline + GifLogger
    """

    def __init__(self, cfg: Dict[str, Any], memory, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self._initialized = False

        self._pipeline: Optional[PipelineExecutor] = None
        self._evt: Optional[EventService] = None

        zm_cfg = self.cfg.get("zero_model", {}) or {}
        self._gif_fps: int = int(zm_cfg.get("fps", 8))
        self._max_frames: int = int(zm_cfg.get("max_frames", 1024))
        self._out_dir: str = (
            self.cfg.get("paths", {}).get("vpm_out")
            or zm_cfg.get("output_dir")
            or "data/vpms"
        )
        os.makedirs(self._out_dir, exist_ok=True)

        # state
        self._sessions: Dict[str, _TimelineSession] = {}
        self._bus_bindings: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ #
    # SERVICE PROTOCOL
    # ------------------------------------------------------------------ #
    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return
        pipeline_cfg = self.cfg.get("zero_model", {}).get("pipeline") or _DEFAULT_PIPELINE
        self._pipeline = PipelineExecutor(pipeline_cfg)
        self._evt = EventService(self.cfg, self.memory, self.logger)
        self._evt.initialize()
        self._initialized = True
        log.debug("ZeroModelService initialized")

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "active_sessions": len(self._sessions),
                "fps": self._gif_fps,
                "max_frames": self._max_frames,
                "out_dir": self._out_dir,
            },
        }

    def shutdown(self) -> None:
        self._sessions.clear()
        self._bus_bindings.clear()
        self._pipeline = None
        self._initialized = False
        log.debug("ZeroModelService shutdown")

    @property
    def name(self) -> str:
        return "zeromodel-service-v2"

    # ------------------------------------------------------------------ #
    # TIMELINE CONTROL
    # ------------------------------------------------------------------ #
    def timeline_open(
        self,
        run_id: str,
        *,
        metrics: Optional[List[str]] = None,
        out_dir: Optional[str] = None,
    ) -> None:
        if not run_id:
            raise ValueError("timeline_open requires run_id")
        if run_id in self._sessions:
            return

        odir = out_dir or self._out_dir
        os.makedirs(odir, exist_ok=True)

        # ✅ Leave metrics_order empty — will be set on first append
        self._sessions[run_id] = _TimelineSession(
            run_id=run_id,
            metrics_order=list(metrics) if metrics else [],  # store if caller passed explicit order
            out_dir=odir,
        )
        log.debug(f"Timeline opened for run_id={run_id}")

    def timeline_append_row(
        self,
        run_id: str,
        *,
        metrics_columns: List[str],
        metrics_values: List[Any],
    ) -> None:
        sess = self._sessions.get(run_id)
        if not sess:
            self._sessions[run_id] = _TimelineSession(
                run_id=run_id,
                metrics_order=list(metrics_columns),
                out_dir=self._out_dir,
            )
            sess = self._sessions[run_id]

        # ----- validate inputs -----
        if not isinstance(metrics_columns, list) or not isinstance(metrics_values, list):
            log.warning("[ZeroModelService] timeline_append_row: bad types for columns/values")
            return
        if len(metrics_columns) != len(metrics_values):
            log.warning(
                "[ZeroModelService] timeline_append_row: length mismatch cols=%d vals=%d",
                len(metrics_columns), len(metrics_values)
            )
            # best-effort truncate to min length
            n = min(len(metrics_columns), len(metrics_values))
            metrics_columns = metrics_columns[:n]
            metrics_values  = metrics_values[:n]

        # ----- scale config (use the same place you read in initialize) -----
        scale_mode = (
            self.cfg.get("timeline_scale_mode")
            or self.cfg.get("zero_model", {}).get("timeline_scale_mode")
            or "passthrough"
        ).lower()

        # ----- build normalized values + mapping -----
        normalized: List[float] = []
        name_to_val: Dict[str, float] = {}

        for c, v in zip(metrics_columns, metrics_values):
            try:
                f = float(v)
            except Exception:
                f = 0.0
            if not np.isfinite(f):
                f = 0.0

            f_scaled = f
            if scale_mode == "clip01":
                f_scaled = max(0.0, min(1.0, f))
            elif scale_mode == "percent_0_100":
                f_scaled = max(0.0, min(1.0, f / 100.0))
            elif scale_mode == "robust01":
                # defer robust scaling to finalize; store raw here
                pass  # keep f_scaled = f
            # else: passthrough

            f_scaled = float(f_scaled)
            normalized.append(f_scaled)
            name_to_val[str(c)] = f_scaled

        # ----- initialize or expand metric order -----
        if not sess.metrics_order:
            sess.metrics_order = list(metrics_columns)
            log.debug("[ZeroModelService] Metric order initialized → %s", sess.metrics_order)
        if len(metrics_columns) != len(sess.metrics_order):
            expected_set = set(sess.metrics_order)
            received_set = set(metrics_columns)
            log.warning(
                "[ZeroModelService] Mismatched metrics for run_id=%s (expected=%d got=%d) "
                "Missing=%s Extra=%s",
                run_id, len(sess.metrics_order), len(metrics_columns),
                list(expected_set - received_set) or "None",
                list(received_set - expected_set) or "None",
            )

        # Add any new columns that appear later
        if len(metrics_columns) > len(sess.metrics_order):
            for name in metrics_columns:
                if name not in sess.metrics_order:
                    sess.metrics_order.append(name)
            # pad older rows to new width
            for i in range(len(sess.rows)):
                sess.rows[i] += [0.0] * (len(sess.metrics_order) - len(sess.rows[i]))

        # ----- align and append row -----
        row = [float(name_to_val.get(name, 0.0)) for name in sess.metrics_order]

        # make 'normalized' from metrics_values
        normalized = []
        for v in metrics_values:
            try:
                f = float(v)
                if not np.isfinite(f): f = 0.0
            except Exception:
                f = 0.0
            # optional scaling switch here...
            normalized.append(float(f))

        # then map names -> normalized values
        name_to_val = dict(zip(metrics_columns, normalized))
        row = [float(name_to_val.get(name, 0.0)) for name in sess.metrics_order]
        sess.rows.append(row)

        # quick per-row stats (sanity)
        try:
            nz = sum(1 for x in row if x != 0.0)
            log.debug(
                "[ZeroModelService] row appended: cols=%d nonzero=%d min=%.4f max=%.4f mean=%.4f",
                len(row), nz, float(min(row)), float(max(row)), float(sum(row)/max(1,len(row)))
            )
        except Exception:
            pass


    # ------------------------------------------------------------------ #
    # FINALIZE & RENDER
    # ------------------------------------------------------------------ #
    async def timeline_finalize(
        self,
        run_id: str,
        *,
        fps: Optional[int] = None,
        datestamp: bool = True,
        out_path: Optional[str] = "data/vpms",
        progress_cb: Optional[Callable[[str, int, int], None]] = None, 
    ) -> Dict[str, Any]:
        if progress_cb: progress_cb("finalize:start", 0, 1)
        log.debug(f"[ZeroModelService] Finalizing timeline for run_id={run_id}")
        sess = self._sessions.pop(run_id, None)
        if not sess:
            return {"status": "noop", "reason": "no_session"}

        mat = sess.as_matrix()

        # 🔧 Apply requested timeline scaling here (this was previously only "planned")
        scale_mode = (self.cfg.get("timeline_scale_mode") or "robust01").lower()
        if scale_mode == "robust01":
            mat = _robust_scale_cols(mat, lo_p=1.0, hi_p=99.0)
        elif scale_mode == "percent_0_100":
            # treat values like "0..100" percentages → normalize to 0..1, clamp
            mat = np.clip(np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0) / 100.0, 0.0, 1.0)
        else:
            # passthrough: at least clean NaN/Inf to zeros
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

        log.info(
            "[ZeroModelService] finalize scale=%s | shape=%s | min=%.4f max=%.4f mean=%.4f",
            scale_mode, mat.shape, float(mat.min()), float(mat.max()), float(mat.mean())
        )

        if mat.shape[0] < 3 or mat.shape[1] < 2:
            log.warning(f"[ZeroModelService] Too few rows ({mat.shape}) for run_id={run_id}, deferring finalize.")
            # Keep session alive for a few more seconds
            self._sessions[run_id] = sess
            await asyncio.sleep(4.0)
            mat = sess.as_matrix()

        if mat.size == 0 or mat.shape[0] < 2:
            log.warning(f"[ZeroModelService] Empty or too small matrix for run_id={run_id}, skipping render.")
            return {"status": "empty", "matrix": mat}

        log.debug(
            f"ZeroModelService: finalizing timeline for run_id={run_id} "
            f"with {mat.shape[0]} steps and {mat.shape[1]} metrics"
        )

        # Timestamped output dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_prefix = str(run_id)[:8]
        run_dir = os.path.join(out_path, f"run_{run_prefix}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        base_name = f"OK{run_prefix}_{timestamp}"
        gif_path = os.path.join(run_dir, base_name + ".gif")

        # Render animated timeline GIF
        if progress_cb: progress_cb("timeline:render:start", 0, int(mat.shape[0]))
        res = self.render_timeline_from_matrix(
            mat,
            gif_path,
            fps=(fps or self._gif_fps),
            metrics=sess.metrics_order,
            options={"panel": "timeline"},
            datestamp=datestamp,
        )
        if progress_cb: progress_cb("timeline:render:done", int(mat.shape[0]), int(mat.shape[0]))

        # Write meta JSON
        meta_path = os.path.join(run_dir, base_name + ".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(dumps_safe(
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "metrics": sess.metrics_order,
                    "shape": res["shape"],
                },
                indent=2
            ))

        # Render static summary PNG
        summary_path = self.render_static_summary(
            mat,
            out_path=run_dir,
            metrics=sess.metrics_order,
            label="timeline",
            timestamp=timestamp,
        )
        log.debug(f"ZeroModelService: summary image saved → {summary_path}")


        # ------------------------------------------------------------------
        # 🌌 Auto-generate epistemic field (optional)
        # ------------------------------------------------------------------
        try:
            # Build small contrastive populations from the current matrix
            # Here we treat the first half as "positive" and second half as "negative"
            if mat.size >= 4 and mat.shape[0] > 2:
                if progress_cb: progress_cb("ef:prepare", 0, 1)
                midpoint = mat.shape[0] // 2
                pos_mats = [mat[:midpoint, :]]
                neg_mats = [mat[midpoint:, :]]

                metric_names = (
                    sess.metrics_order
                    if sess and getattr(sess, "metrics_order", None)
                    else [f"metric_{i}" for i in range(mat.shape[1])]
                )
                field_meta = self.generate_epistemic_field(
                    pos_matrices=pos_mats,
                    neg_matrices=neg_mats,
                    output_dir=os.path.join(run_dir, "epistemic_fields"),
                    metric_names=metric_names,
                )
                if progress_cb: progress_cb("ef:done", 1, 1)
                log.debug(
                    f"[ZeroModelService] 🧠 Epistemic field auto-generated "
                    f"(ΔMass={field_meta['delta_mass']:.4f}) → {field_meta['png']}"
                )
            else:
                log.warning("[ZeroModelService] Matrix too small for epistemic field generation.")
        except Exception as e:
            log.warning("[ZeroModelService] Epistemic field generation failed: %s", e)


        # Publish event
        if self._evt:
            await self._evt.publish(
                "arena.ats.timeline_ready",
                {
                    "run_id": run_id,
                    "gif": res["output_path"],
                    "json": meta_path,
                    "png": summary_path,
                    "steps": mat.shape[0],
                    "metrics": sess.metrics_order,
                },
            )
        if progress_cb: progress_cb("finalize:done", 1, 1)
        return {
            "status": "ok",
            "matrix": mat,
            "metric_names": sess.metrics_order,
            **res,
            "meta_path": meta_path,
            "summary_path": summary_path,
        }


    def build_intensity_report(
        self,
        *,
        hrm_matrix: np.ndarray,
        tiny_matrix: np.ndarray,
        hrm_metric_names: list[str],
        tiny_metric_names: list[str],
        out_dir: str,
        top_k: int = 20,
    ) -> dict:
        """
        Create JSON summaries for HRM, Tiny, and Diff (HRM−Tiny):
        - top columns (by mean |intensity|)
        - top rows (by mean |intensity|)
        - cross-model comparison on common canonical dims
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # 1) Per-model intensity
        col_h = _column_intensity(hrm_matrix)
        col_t = _column_intensity(tiny_matrix)
        row_h = _row_intensity(hrm_matrix)
        row_t = _row_intensity(tiny_matrix)

        h_idx, h_ranked = _rank_intensity(col_h, hrm_metric_names)
        t_idx, t_ranked = _rank_intensity(col_t, tiny_metric_names)

        top_cols_hrm = h_ranked[:top_k]
        top_cols_tiny = t_ranked[:top_k]

        top_rows_hrm = np.argsort(-row_h)[:top_k].tolist()
        top_rows_tiny = np.argsort(-row_t)[:top_k].tolist()

        # 2) Diff field aligned on shared shape (rows already aligned by build)
        r = min(hrm_matrix.shape[0], tiny_matrix.shape[0])
        c = min(hrm_matrix.shape[1], tiny_matrix.shape[1])
        D = np.asarray(hrm_matrix[:r, :c] - tiny_matrix[:r, :c], dtype=np.float64)
        col_d = _column_intensity(D)
        d_idx, d_ranked = _rank_intensity(col_d, hrm_metric_names[:c])  # use HRM names slice
        top_cols_diff = d_ranked[:top_k]
        top_rows_diff = np.argsort(-_row_intensity(D))[:top_k].tolist()

        # 3) Cross-model comparison on canonical dimension keys
        canon_h = [_canonical_key(n) for n in hrm_metric_names]
        canon_t = [_canonical_key(n) for n in tiny_metric_names]
        common = sorted(set(canon_h) & set(canon_t))
        # Build per-dimension aggregates (mean intensity for all columns mapping to that dim)
        def _agg_by_dim(names, intensities):
            grp = defaultdict(list)
            for n, v in zip(names, intensities):
                grp[_canonical_key(n)].append(float(v))
            return {k: float(np.mean(vs)) for k, vs in grp.items()}
        agg_h = _agg_by_dim(hrm_metric_names, col_h)
        agg_t = _agg_by_dim(tiny_metric_names, col_t)

        comp = []
        for d in common:
            comp.append({
                "dimension": d,
                "hrm_mean_abs": agg_h.get(d, 0.0),
                "tiny_mean_abs": agg_t.get(d, 0.0),
                "delta": float(agg_h.get(d, 0.0) - agg_t.get(d, 0.0)),
                "ratio": float((agg_h.get(d, 1e-12)) / (agg_t.get(d, 1e-12))),
            })
        # rank-based similarity (optional)
        try:
            # ranks over common dims
            rh = np.array([agg_h[d] for d in common])
            rt = np.array([agg_t[d] for d in common])
            kendall = float(spstats.kendalltau(rh, rt).correlation) if len(common) >= 2 else None
            spearman = float(spstats.spearmanr(rh, rt).correlation) if len(common) >= 2 else None
        except Exception:
            kendall = spearman = None

        report = {
            "summary": {
                "rows": r, "hrm_cols": len(hrm_metric_names), "tiny_cols": len(tiny_metric_names),
                "top_k": top_k,
                "rank_corr": {"kendall_tau": kendall, "spearman_rho": spearman},
            },
            "hrm": {
                "top_columns": top_cols_hrm,
                "top_rows": top_rows_hrm,
            },
            "tiny": {
                "top_columns": top_cols_tiny,
                "top_rows": top_rows_tiny,
            },
            "diff": {
                "top_columns": top_cols_diff,
                "top_rows": top_rows_diff,
            },
            "by_dimension_common": sorted(comp, key=lambda x: -abs(x["delta"])),
        }

        out_path = Path(out_dir) / "intensity_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(dumps_safe(report, indent=2))

        return {"path": str(out_path), **report}

    # ------------------------------------------------------------------ #
    # RENDER HELPERS
    # ------------------------------------------------------------------ #
    def render_timeline_from_matrix(
        self,
        matrix: np.ndarray,
        out_path: str,
        *,
        fps: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        datestamp: bool = False,
        progress_cb: Optional[Callable[[str,int,int],None]] = None,
    ) -> Dict[str, Any]:
        assert self._pipeline is not None, "ZeroModelService not initialized"

        if not isinstance(matrix, np.ndarray) or matrix.size == 0:
            ncols = 1 if metrics is None else max(1, len(metrics))
            matrix = np.zeros((1, ncols), dtype=np.float32)

        gif = GifLogger(max_frames=self._max_frames)
        fps = fps or self._gif_fps

        # keep natural time order; column-0 sort can destroy contrast and order
        M = np.asarray(matrix, dtype=np.float32)
        total = int(M.shape[0])
        for i in range(total):
            row = M[i : i + 1, :]
            vpm_out, _ = self._pipeline.run(row, {"enable_gif": False})

        # sorted_matrix = self.sort_on_first_index(matrix)
        # total = int(sorted_matrix.shape[0])
        # for i in range(total):
        #     row = sorted_matrix[i : i + 1, :]
        #     vpm_out, _ = self._pipeline.run(row, {"enable_gif": False})
            gif.add_frame(
                vpm_out,
                metrics={
                    "step": i,
                    "loss": 1 - row.mean(),
                    "val_loss": row.mean(),
                    "acc": row.std(),
                },
            )
            if progress_cb and ((i % 10) == 0 or (i + 1) == total):
                progress_cb("timeline:frame", i + 1, total)
            if (i % 25) == 0 or (i + 1) == total:
                log.debug("[ZeroModelService] frame %d/%d stats: min=%.4f max=%.4f mean=%.4f",
                    i+1, total, float(row.min()), float(row.max()), float(row.mean()))


        gif.save_gif(out_path, fps=fps)
        log.debug(f"ZeroModelService: rendered {len(gif.frames)} frames → {out_path}")

        return {"output_path": out_path, "frames": len(gif.frames), "shape": list(matrix.shape)}

    def render_static_summary(
        self,
        matrix: np.ndarray,
        out_path: str,
        *,
        metrics: Optional[List[str]] = None,
        label: Optional[str] = None,
        cmap: str = "viridis",
        timestamp: Optional[str] = None,
    ) -> str:
        """Render a static heatmap summary of all steps × metrics."""
        if not isinstance(matrix, np.ndarray) or matrix.size == 0:
            log.warning("render_static_summary called with empty matrix")
            return out_path

        try:
            ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            label_part = f"_{label}" if label else ""
            base_name = f"vpm_summary{label_part}_{ts}.png"
            png_path = os.path.join(out_path, base_name)

            fig, ax = plt.subplots(
                figsize=(max(6, matrix.shape[1] / 4), max(4, matrix.shape[0] / 20))
            )
            im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

            if metrics:
                ax.set_xticks(range(len(metrics)))
                ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
            ax.set_yticks([])
            ax.set_title(f"VPM Summary — {label or 'Run'} ({ts})", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)

            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close(fig)

            log.debug(f"ZeroModelService: static VPM summary saved → {png_path}")
            return png_path
        except Exception as e:
            log.error(f"render_static_summary failed: {e}")
            return out_path

    def sort_on_first_index(self, matrix: np.ndarray, descending: bool = False) -> np.ndarray:
        """
        Sort a 2D NumPy matrix by its first column (index 0).
        Designed for Stephanie VPM matrices where the first column is node_id.

        Args:
            matrix (np.ndarray): The input 2D array (N x D).
            descending (bool): Sort descending if True (default False).

        Returns:
            np.ndarray: Sorted matrix (same shape as input).

        Behavior:
            - Skips sorting if the matrix has < 2 rows or < 1 column.
            - Casts first column to float for safety.
            - Logs before/after ranges for debugging.
        """
        try:
            if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
                return matrix

            # Defensive cast for mixed types
            col0 = matrix[:, 0].astype(float)
            order = np.argsort(col0)
            if descending:
                order = order[::-1]

            sorted_matrix = matrix[order]
            log.debug(
                f"Matrix sorted by first index "
                f"(min={col0.min():.4f}, max={col0.max():.4f}, rows={matrix.shape[0]})"
            )
            return sorted_matrix

        except Exception as e:
            log.warning("[ZeroModelService] sort_on_first_index failed: %s", e)
            return matrix



    def generate_epistemic_field(
        self,
        pos_matrices: list[np.ndarray],
        neg_matrices: list[np.ndarray],
        output_dir: str = "data/epistemic_field",
        alpha: float = 0.97,
        pos_label: str = "HRM",
        neg_label: str = "Tiny",
        Kc: int = 40,
        Kr: int = 100,
        fps: int = 8,
        cmap: str = "seismic",
        aggregate: bool = False,
        metric_names: Optional[List[str]] = None,
        save_individual: bool = True, 
        progress_cb: Optional[Callable[[str,int,int],None]] = None,
    ) -> dict:
        """
        Generate the ZeroModel Epistemic Field — a contrastive visual intelligence map.

        Features:
        - Learns canonical spatial layout from positive samples
        - Projects negative samples into same layout
        - Combines them into a single differential VPM
        - Computes epistemic field overlap (coherence)
        - Renders static + animated representations
        """

        metric_names = list(metric_names or [])
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(output_dir, f"epistemic_field_{ts}")

        # -------------------------------
        # 1️⃣ Aggregate or stack matrices
        # -------------------------------
        if progress_cb: progress_cb("stack", 1, 5)
        def _normalize_field(matrix: np.ndarray) -> np.ndarray:
            matrix = np.nan_to_num(matrix)
            max_val = np.max(np.abs(matrix)) + 1e-8
            return matrix / max_val

        def _align_shapes(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Crop or pad matrices to smallest common shape."""
            rows = min(A.shape[0], B.shape[0])
            cols = min(A.shape[1], B.shape[1])
            return A[:rows, :cols], B[:rows, :cols]

        # 1️⃣ Aggregate or stack
        if aggregate:
            # Mean over multiple samples (population field)
            X_pos = np.mean(np.stack(pos_matrices, axis=0), axis=0)
            X_neg = np.mean(np.stack(neg_matrices, axis=0), axis=0)
        else:
            # Simple vertical stack
            X_pos = np.vstack(pos_matrices)
            X_neg = np.vstack(neg_matrices)

        # --- early exit for empty or degenerate matrices ---
        if X_pos.size == 0 or X_neg.size == 0 or X_pos.shape[0] < 2 or X_neg.shape[0] < 2:
            log.warning("[ZeroModelService] Skipping epistemic field generation — insufficient data.")
            return {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "mass_pos": 0.0,
                "mass_neg": 0.0,
                "delta_mass": 0.0,
                "overlap_score": 0.0,
                "metric_names": metric_names,
                "png": None,
                "gif": None,
            }

        # -------------------------------
        # 2️⃣ Learn canonical layout
        # -------------------------------
        if progress_cb: progress_cb("optimize_layout_start", 2, 5)
        opt = SpatialOptimizer(Kc=Kc, Kr=Kr, alpha=alpha)
        opt.apply_optimization([X_pos])
        w = opt.metric_weights
        layout = opt.canonical_layout
        if progress_cb: progress_cb("optimize_layout_done", 3, 5)

        # Build index mapping: old_index -> new_position (based on canonical layout)
        if layout is not None:
            flat_layout = np.ravel(layout)
            order = np.argsort(flat_layout)
            if metric_names and len(order) > len(metric_names):
                order = order[: len(metric_names)]
            index_mapping = {orig_idx: int(new_idx) for new_idx, orig_idx in enumerate(order)}
        else:
            index_mapping = {i: i for i in range(len(metric_names or []))}



        # -------------------------------
        # 3️⃣ Apply canonical transform
        # -------------------------------
        if progress_cb: progress_cb("project", 4, 5)
        Y_pos, _, _ = opt.phi_transform(X_pos, w, w)
        Y_neg, _, _ = opt.phi_transform(X_neg, w, w)

        # Normalize & align
        Y_pos = _normalize_field(Y_pos)
        Y_neg = _normalize_field(Y_neg)
        Y_pos, Y_neg = _align_shapes(Y_pos, Y_neg)

        # -------------------------------
        # 4️⃣ Differential & metrics
        # -------------------------------
        diff = Y_pos - Y_neg
        mass_pos = opt.top_left_mass(Y_pos)
        mass_neg = opt.top_left_mass(Y_neg)
        delta_mass = mass_pos - mass_neg

        # Epistemic overlap (structural coherence)
        overlap = float(np.sum(np.minimum(Y_pos, Y_neg)) / (np.sum(np.maximum(Y_pos, Y_neg)) + 1e-8))

        log.debug(f"Epistemic field generated: +mass={mass_pos:.4f}, -mass={mass_neg:.4f}, Δ={delta_mass:.4f}, overlap={overlap:.4f}")

        # ================================================================
        # 5️⃣ Visualization — side-by-side comparison of transformation steps
        # ================================================================

        def _make_visual_grid(images: List[np.ndarray], titles: List[str], base_path: str):
            """Render and save a static grid comparison image."""
            fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
            for ax, img, title in zip(axes, images, titles):
                im = ax.imshow(img, cmap=cmap, aspect="auto")
                ax.set_title(title, fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
                ax.axis("off")
            plt.tight_layout()
            comp_path = base_path + "_comparison.png"
            plt.savefig(comp_path, dpi=150)
            plt.close(fig)
            return comp_path


        def _make_transition_gif(stages: List[np.ndarray], titles: List[str], gif_path: str):
            """Create animated GIF showing step-by-step transformation (fully in-memory)."""
            giflog = GifLogger(max_frames=100)

            for i, (img, title) in enumerate(zip(stages, titles)):
                # Create a matplotlib figure for the current stage
                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(img, cmap=cmap, aspect="auto")
                ax.set_title(title, fontsize=12)
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
                plt.tight_layout()

                # 🧠 Convert the rendered plot to an RGB NumPy array (backend-safe)
                fig.canvas.draw()

                try:
                    # Modern Matplotlib (>=3.8)
                    buf = np.asarray(fig.canvas.buffer_rgba())
                except AttributeError:
                    # Older Matplotlib (<3.8)
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # Ensure 3D RGB array
                frame = np.array(buf, copy=True)
                giflog.add_frame(frame, metrics={"stage": title})
                plt.close(fig)

            giflog.save_gif(gif_path, fps=1)
            log.debug(f"Transformation GIF saved → {gif_path}")

        def _save_single(img: np.ndarray, title: str, base_path: str) -> str:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(img, cmap=cmap, aspect="auto")
            ax.set_title(title, fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
            ax.axis("off")
            plt.tight_layout()
            out_path = f"{base_path}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path

        reordered_metric_names = []
        if metric_names:
            for old_idx in range(len(metric_names)):
                new_idx = index_mapping.get(old_idx, old_idx)
                if new_idx < len(metric_names):
                    reordered_metric_names.append(metric_names[new_idx])
                else:
                    reordered_metric_names.append(metric_names[old_idx])
        else:
            reordered_metric_names = [f"metric_{i}" for i in range(diff.shape[1])]

        log.debug(f"[ZeroModelService] Metric reordering preserved {len(reordered_metric_names)} names")

        # ================================================================
        # 8️⃣ Subfield Extraction — Q(5) vs E(5) + Overlay (SCM-aware, HRM/Tiny fallback)
        # ================================================================
        try:
            def _robust01(a: np.ndarray, lo=1, hi=99) -> np.ndarray:
                a = np.asarray(a, dtype=np.float32)
                if a.size == 0: return a
                lo_v, hi_v = np.percentile(a, [lo, hi])
                if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v <= lo_v:
                    return np.zeros_like(a)
                x = (a - lo_v) / (hi_v - lo_v)
                return np.clip(x, 0.0, 1.0).astype(np.float32)

            # Helpers to find per-dimension "score-like" cols, tolerant of hrm./tiny. prefixes
            CAND_SUFFIXES = ("", ".score", ".aggregate", ".raw", ".value")
            def _pick_dim_col(names: list[str], dim: str) -> int | None:
                # exact SCM first
                scm_exact = f"scm.{dim}.score01"
                low = [n.lower() for n in names]
                if scm_exact in low:
                    return low.index(scm_exact)
                # hrm./tiny. style
                for i, n in enumerate(low):
                    if f".{dim}" in n and n.endswith(CAND_SUFFIXES):
                        return i
                # looser fallback: any column that contains the dim token
                for i, n in enumerate(low):
                    if f".{dim}" in n:
                        return i
                return None

            dims5 = ["reasoning","knowledge","clarity","faithfulness","coverage"]
            # 1) Build Q: try SCM first; otherwise per-dimension score columns
            q_idxs = []
            for d in dims5:
                idx = _pick_dim_col(reordered_metric_names, d)
                if idx is not None: q_idxs.append(idx)

            if len(q_idxs) == 0:
                log.warning("[ZeroModelService] No Q columns found; skipping Q/E overlays.")
                q_path = e_path = o_path = None
            else:
                q_field = diff[:, q_idxs]                 # shape (T, k_q)
                # ensure 2D
                if q_field.ndim == 1:
                    q_field = q_field.reshape(-1, 1)

                # 2) Build E buckets to 5 dims (configurable; tolerant lookups)
                E_BUCKETS = {
                    "reasoning":    ["scm.consistency01", "scm.temp01"],
                    "knowledge":    ["scm.ood_hat01", "recon_sim"],
                    "clarity":      ["scm.length_norm01", "aux3_p_bad"],
                    "faithfulness": ["recon_sim", "scm.consistency01"],
                    "coverage":     ["concept_sparsity", "scm.uncertainty01"],
                }
                name2idx = {n.lower(): i for i, n in enumerate(reordered_metric_names)}
                low_names = [n.lower() for n in reordered_metric_names]

                def _find_any(indices_or_names: list[str]) -> list[int]:
                    out = []
                    for key in indices_or_names:
                        k = key.lower()
                        if k in name2idx:
                            out.append(name2idx[k]); continue
                        # fuzzy: match if all parts appear
                        parts = k.split(".")
                        for i, n in enumerate(low_names):
                            if all(p in n for p in parts):
                                out.append(i); break
                    return sorted(set(out))

                e_cols = []
                for d in dims5:
                    idxs = _find_any(E_BUCKETS.get(d, []))
                    if not idxs:
                        e_cols.append(None)
                    else:
                        e_cols.append(idxs)

                # E as (T, len(dims5)), averaging each bucket (or 0 if missing)
                T = diff.shape[0]
                e_field = np.zeros((T, len(dims5)), dtype=np.float32)
                for j, idxs in enumerate(e_cols):
                    if idxs:
                        e_field[:, j] = np.mean(diff[:, idxs], axis=1)

                # If Q has k!=5 columns, align to 5 by projecting/tiling
                if q_field.shape[1] != len(dims5):
                    # project Q into 5 via nearest-dim mapping: take first k dims we have, pad rest with zeros
                    q5 = np.zeros((T, len(dims5)), dtype=np.float32)
                    k = min(q_field.shape[1], len(dims5))
                    q5[:, :k] = q_field[:, :k]
                    q_field = q5

                # 3) Robust scales
                q_field_norm = _robust01(q_field)
                e_field_norm = _robust01(e_field)

                # 4) Overlay (shapes guaranteed compatible)
                overlay = q_field_norm - e_field_norm

                q_path = _save_field_image(q_field_norm, "Q-Field (5D or fewer)", f"{base}_q_field")
                e_path = _save_field_image(e_field_norm, "E-Field (5D)",         f"{base}_energy_field")
                o_path = _save_field_image(overlay,       "Q–E Overlay (Q−E)",   f"{base}_overlay")

                # optional correlation
                corr = float(np.corrcoef(q_field_norm.ravel(), e_field_norm.ravel())[0,1]) if q_field_norm.size and e_field_norm.size else None
                log.debug(f"[ZeroModelService] Subfields saved → Q:{bool(q_path)} E:{bool(e_path)} Overlay:{bool(o_path)} Corr={corr}")

        except Exception as e:
            log.warning("[ZeroModelService] Subfield extraction failed: %s", e)


        # ------------------------------- 
        # Prepare visual stages
        # -------------------------------
        raw_good = _normalize_field(X_pos)
        raw_bad  = _normalize_field(X_neg)
        opt_good = Y_pos
        opt_bad  = Y_neg
        combined = diff

        titles = [
            f"Raw {pos_label} (Before Optimization)",
            f"Optimized {pos_label} (Spatial Calculus)",
            f"Raw {neg_label} (Before Optimization)",
            f"Optimized {neg_label} (Spatial Calculus)",
            f"Differential Field ({pos_label} − {neg_label})",
        ]
        images = [raw_good, opt_good, raw_bad, opt_bad, combined]

        # Render comparison grid + GIF
        comparison_path = _make_visual_grid(images, titles, base)
        transition_gif_path = base + "_transform.gif"
        _make_transition_gif(images, titles, transition_gif_path)
        log.debug(f"Epistemic field visual comparison saved → {comparison_path}")


        single_paths = {}
        if save_individual:
            single_paths["raw_pos_png"] = _save_single(raw_good,  f"Raw {pos_label}",              base + "_raw_pos")
            single_paths["opt_pos_png"] = _save_single(opt_good,  f"Optimized {pos_label}",         base + "_opt_pos")
            single_paths["raw_neg_png"] = _save_single(raw_bad,   f"Raw {neg_label}",               base + "_raw_neg")
            single_paths["opt_neg_png"] = _save_single(opt_bad,   f"Optimized {neg_label}",         base + "_opt_neg")
            single_paths["diff_png"]    = _save_single(combined,  f"Differential ({pos_label}−{neg_label})", base + "_diff")


        # -------------------------------
        # 5️⃣ Render static field image
        # -------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(diff, cmap=cmap, aspect="auto")
        ax.set_title(f"ZeroModel Epistemic Field — ΔMass={delta_mass:.4f}, Overlap={overlap:.4f}")
        fig.colorbar(im, ax=ax, label="Δ Intensity (Pos−Neg)")
        plt.tight_layout()
        png_path = base + ".png"
        plt.savefig(png_path, dpi=150)
        plt.close(fig)

        # -------------------------------
        # 6️⃣ Optional animated timeline
        # -------------------------------
        giflog = GifLogger(max_frames=300)
        pipeline = PipelineExecutor([
            {"stage": "normalize", "params": {}},
            {"stage": "organization", "params": {"strategy": "spatial"}},
        ])

        for i in range(diff.shape[0]):
            row = diff[i:i+1, :]
            out, _ = pipeline.run(row, {"enable_gif": False})
            giflog.add_frame(out, metrics={"step": i, "mass": float(row.mean())})

        gif_path = base + ".gif"
        giflog.save_gif(gif_path, fps=fps)

        # -------------------------------
        # 7️⃣ Persist metadata
        # -------------------------------
        meta = {
            "timestamp": ts,
            "mass_pos": mass_pos,
            "mass_neg": mass_neg,
            "delta_mass": delta_mass,
            "overlap_score": overlap,
            "comparison_png": comparison_path,
            "transform_gif": transition_gif_path,
            "metric_weights": w.tolist(),
            "canonical_layout": layout.tolist(),
            "metric_index_mapping": index_mapping,
            "metric_names_original": metric_names,
            "metric_names_reordered": reordered_metric_names,
            "timeline_scale_mode": (self.cfg.get("timeline_scale_mode") or "robust01"),
            "png": base + ".png",
            "gif": gif_path,
            "diff_matrix": diff.tolist(),
            "metric_names": reordered_metric_names,
            **single_paths,  
        }

        meta_path = base + ".json"
        text=dumps_safe(meta)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(text)

        log.debug(f"Epistemic field saved → {meta_path}")
        return meta


    def generate_epistemic_field_phos_ordered(
        self,
        pos_matrices: list[np.ndarray],
        neg_matrices: list[np.ndarray],
        output_dir: str = "data/epistemic_field",
        *,
        alpha: float = 0.97,
        pos_label: str = "HRM",
        neg_label: str = "Tiny",
        iters: int = 4,                 # number of row/col reorder passes
        cmap: str = "seismic",
        aggregate: bool = False,
        metric_names: Optional[List[str]] = None,
        progress_cb: Optional[Callable[[str,int,int],None]] = None,
    ) -> dict:
        """
        PHOS-style 2D ordering: reorder columns and rows to concentrate intensity
        in the top-left, then compare pos vs neg using the *same* order.

        This removes the vertical striping look by destroying raw column identity
        in favor of a canonical, high-mass-first layout.
        """
        metric_names = list(metric_names or [])
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(output_dir, f"epistemic_field_phos_{ts}")

        def _robust(X: np.ndarray) -> np.ndarray:
            X = np.nan_to_num(np.asarray(X, dtype=np.float64))
            if X.size == 0:
                return X
            # robust scale per-matrix
            lo = np.percentile(X, 10)
            hi = np.percentile(X, 90)
            if hi <= lo:
                hi = lo + 1.0
            Y = (X - lo) / (hi - lo)
            return np.clip(Y, 0.0, 1.0)

        def _stack(mats: list[np.ndarray]) -> np.ndarray:
            return np.mean(np.stack(mats, axis=0), axis=0) if aggregate else np.vstack(mats)

        # 1) Build matrices
        X_pos = _stack(pos_matrices)
        X_neg = _stack(neg_matrices)
        if X_pos.size == 0 or X_neg.size == 0 or X_pos.ndim != 2 or X_neg.ndim != 2:
            log.warning("[ZeroModelService] PHOS-ordered: insufficient data.")
            return {"status": "empty"}

        # Align shapes conservatively
        r = min(X_pos.shape[0], X_neg.shape[0])
        c = min(X_pos.shape[1], X_neg.shape[1])
        X_pos = X_pos[:r, :c]
        X_neg = X_neg[:r, :c]

        # 2) Normalize
        A = _robust(X_pos)  # use POS to learn the ordering
        B = _robust(X_neg)

        # 3) Iterative TL ordering (columns then rows by mean |value|)
        row_order = np.arange(A.shape[0])
        col_order = np.arange(A.shape[1])

        for _ in range(max(1, iters)):
            # columns: sort by mean abs down (more mass first)
            col_scores = np.mean(np.abs(A), axis=0)
            col_order = np.argsort(-col_scores)
            A = A[:, col_order]
            B = B[:, col_order]

            # rows: sort by mean abs down
            row_scores = np.mean(np.abs(A), axis=1)
            row_order = np.argsort(-row_scores)
            A = A[row_order, :]
            B = B[row_order, :]

        Y_pos = A
        Y_neg = B
        diff = Y_pos - Y_neg

        # 4) Quantities
        def _top_left_mass(M: np.ndarray, frac: float = 0.25) -> float:
            s = M.shape[0]
            k = max(1, int(round(np.sqrt(max(frac, 1e-9)) * s)))
            k = min(k, s)
            return float(M[:k, :k].sum()) / (float(M.sum()) + 1e-8)

        mass_pos = _top_left_mass(Y_pos)
        mass_neg = _top_left_mass(Y_neg)
        delta_mass = mass_pos - mass_neg
        overlap = float(np.sum(np.minimum(Y_pos, Y_neg)) / (np.sum(np.maximum(Y_pos, Y_neg)) + 1e-8))

        # 5) Visuals
        def _grid(images, titles, path_base):
            fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
            for ax, img, title in zip(axes, images, titles):
                im = ax.imshow(img, cmap=cmap, aspect="auto")
                ax.set_title(title, fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
                ax.axis("off")
            plt.tight_layout()
            p = path_base + "_comparison.png"
            plt.savefig(p, dpi=150)
            plt.close(fig)
            return p

        comp_titles = [
            f"{pos_label} (PHOS-ordered)",
            f"{neg_label} (PHOS-ordered)",
            f"Differential Field ({pos_label} − {neg_label})",
        ]
        comp_images = [Y_pos, Y_neg, diff]
        comparison_path = _grid(comp_images, comp_titles, base)

        # standalone diff
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(diff, cmap=cmap, aspect="auto")
        ax.set_title(f"PHOS-Ordered Field — ΔMass={delta_mass:.4f}, Overlap={overlap:.4f}")
        fig.colorbar(im, ax=ax, label="Δ Intensity (Pos−Neg)")
        plt.tight_layout()
        png_path = base + ".png"
        plt.savefig(png_path, dpi=150)
        plt.close(fig)

        # 6) Names after column reorder (rows are turns; only columns map to metrics)
        if metric_names and len(metric_names) == c:
            metric_names_reordered = [metric_names[j] for j in col_order]
        else:
            metric_names_reordered = [f"metric_{j}" for j in range(c)]

        meta = {
            "timestamp": ts,
            "mass_pos": float(mass_pos),
            "mass_neg": float(mass_neg),
            "delta_mass": float(delta_mass),
            "overlap_score": float(overlap),
            "png": png_path,
            "comparison_png": comparison_path,
            "diff_matrix": diff.tolist(),
            "metric_names": metric_names_reordered,
            "row_order": row_order.tolist(),
            "col_order": col_order.tolist(),
            "pos_label": pos_label,
            "neg_label": neg_label,
        }
        meta_path = base + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(dumps_safe(meta, indent=2))
        return meta

    def render_frontier_map(
        self,
        A, B,
        out_dir,
        *,
        pos_label="HRM",
        neg_label="Tiny",
        k_latent=20,
        cmap_main="magma",
        cmap_delta="seismic",
        progress_cb: Optional[Callable[[str,int,int],None]] = None,
    ):
        """
        Frontier Map: a single composite figure showing the 'layer between models'.
        Inputs:
        A: [n, dA] HRM timeline matrix
        B: [n, dB] Tiny timeline matrix
        Returns:
        dict with file paths + summary stats
        """
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(out_dir, f"frontier_{ts}")

        # --- 1) Trim to overlapping rows
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        n = min(A.shape[0], B.shape[0])
        if n < 3:
            return {"status":"too_few_rows", "n": int(n)}
        A, B = A[:n], B[:n]

        # --- 2) Shared latent space via CCA-lite
        Wa, Wb, can_corr = _cca_shared_projections(A, B, k=k_latent)
        A_lat = A @ Wa        # [n, k]
        B_lat = B @ Wb        # [n, k]

        # --- 3) “Unexplained” layer (residuals) in the shared space
        # We compute both directions; the 'frontier' is what survives in either.
        resid_A = A_lat - B_lat             # what HRM encodes beyond Tiny
        resid_B = B_lat - A_lat             # what Tiny encodes beyond HRM
        # Per-sample magnitude (row-wise)
        mag_A = np.linalg.norm(resid_A, axis=1, ord=2)  # [n]
        mag_B = np.linalg.norm(resid_B, axis=1, ord=2)  # [n]
        # Per-latent magnitude (column-wise)
        col_A = np.linalg.norm(resid_A, axis=0, ord=2)  # [k]
        col_B = np.linalg.norm(resid_B, axis=0, ord=2)  # [k]

        # --- 4) Normalize for visualization
        A_lat_n = _normalize01(A_lat)
        B_lat_n = _normalize01(B_lat)
        resid_n = _normalize01(A_lat - B_lat)

        # --- 5) PHOS mean images (distributional context)
        A_phos = _phos_mean_image(A)
        B_phos = _phos_mean_image(B)
        A_phos_n = _normalize01(A_phos)
        B_phos_n = _normalize01(B_phos)
        A_phos_n, B_phos_n = _align_square_phos(A_phos_n, B_phos_n)
        diff_phos = A_phos_n - B_phos_n


        # --- 6) Composite figure
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.1], height_ratios=[1.0, 1.0], wspace=0.2, hspace=0.28)

        # (a) PHOS energy maps
        ax1 = fig.add_subplot(gs[0,0]); im1 = ax1.imshow(A_phos_n, cmap=cmap_main, aspect="auto"); ax1.set_title(f"{pos_label} PHOS Energy"); ax1.axis("off"); fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.04)
        ax2 = fig.add_subplot(gs[0,1]); im2 = ax2.imshow(B_phos_n, cmap=cmap_main, aspect="auto"); ax2.set_title(f"{neg_label} PHOS Energy"); ax2.axis("off"); fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.04)
        ax3 = fig.add_subplot(gs[0,2]); im3 = ax3.imshow(diff_phos, cmap=cmap_delta, aspect="auto", vmin=-1, vmax=1); ax3.set_title(f"PHOS Δ ({pos_label} − {neg_label})"); ax3.axis("off"); fig.colorbar(im3, ax=ax3, fraction=0.035, pad=0.04)

        # (b) Latent residual heatmap (the frontier)
        ax4 = fig.add_subplot(gs[1,0]); im4 = ax4.imshow(resid_n, cmap=cmap_delta, aspect="auto", vmin=-1, vmax=1)
        ax4.set_title(f"Shared-Latent Residuals ({pos_label} − {neg_label})")
        ax4.set_xlabel("Latent dimension"); ax4.set_ylabel("Sample (row)")
        fig.colorbar(im4, ax=ax4, fraction=0.035, pad=0.04)

        # (c) Per-sample frontier strength
        ax5 = fig.add_subplot(gs[1,1])
        ax5.plot(mag_A, label=f"{pos_label} unexplained", linewidth=1.0)
        ax5.plot(mag_B, label=f"{neg_label} unexplained", linewidth=1.0)
        ax5.set_title("Frontier Strength per Sample")
        ax5.set_xlabel("Sample index"); ax5.set_ylabel("||residual||₂")
        ax5.legend(loc="upper right", fontsize=8)

        # (d) Per-latent frontier strength + canonical corr
        ax6 = fig.add_subplot(gs[1,2])
        idx = np.arange(resid_n.shape[1])
        ax6.bar(idx - 0.2, col_A / (col_A.max()+1e-8), width=0.4, label=f"{pos_label} resid", alpha=0.8)
        ax6.bar(idx + 0.2, col_B / (col_B.max()+1e-8), width=0.4, label=f"{neg_label} resid", alpha=0.8)
        ax6.set_title("Frontier by Latent Axis")
        ax6.set_xlabel("Latent dimension"); ax6.set_ylabel("Normalized magnitude")
        ax6.legend(loc="upper right", fontsize=8)
        # annotate top CCA correlations for context
        if can_corr is not None and can_corr.size:
            text = "Top canonical corr: " + ", ".join([f"{c:.2f}" for c in can_corr[:5]])
            ax6.text(0.02, 0.98, text, transform=ax6.transAxes, va="top", ha="left", fontsize=8)

        png_path = base + ".png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close(fig)

        # --- 7) Emit summary JSON
        meta = {
            "timestamp": ts,
            "rows_used": int(n),
            "k_latent": int(min(k_latent, A_lat.shape[1])),
            "canonical_corr_top5": (can_corr[:5].tolist() if can_corr is not None and can_corr.size else []),
            "residual_sample_stats": {
                f"{pos_label}_mean": float(np.mean(mag_A)),
                f"{pos_label}_max": float(np.max(mag_A)),
                f"{neg_label}_mean": float(np.mean(mag_B)),
                f"{neg_label}_max": float(np.max(mag_B)),
            },
            "paths": {"png": png_path},
            "notes": "Residuals are A_lat - B_lat; larger magnitude = stronger frontier (unexplained layer).",
        }
        meta_path = base + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return {"status":"ok", **meta}



    def analyze_differential_field(self, diff_matrix: np.ndarray, metric_names: list[str], output_dir: str):
        """
        Analyze the differential field (Good - Bad) to identify surviving high-intensity metrics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if diff_matrix is None:
            log.warning("[PhosAnalyzer] Empty diff_matrix — skipping analysis.")
            return {"ranked_metrics": [], "top_indices": [], "top_rows": []}

        # Ensure NumPy array
        if not isinstance(diff_matrix, np.ndarray):
            diff_matrix = np.asarray(diff_matrix, dtype=np.float32)

        # Defensive shape
        if diff_matrix.ndim < 2:
            diff_matrix = np.expand_dims(diff_matrix, axis=0)

        # 1️⃣ Compute mean absolute intensity per metric (column)
        intensities = np.mean(np.abs(diff_matrix), axis=0)
        if np.max(intensities) == 0:
            log.warning("[PhosAnalyzer] Zero-intensity field — skipping plot.")
            return {"ranked_metrics": [], "top_indices": [], "top_rows": []}

        # 2️⃣ Normalize to [0, 1]
        norm_intensities = intensities / np.max(intensities)

        # 3️⃣ Rank descending
        sorted_idx = np.argsort(norm_intensities)[::-1]
        ranked_metrics = [
            {
                "metric": metric_names[i] if i < len(metric_names) else f"metric_{i}",
                "mean_intensity": float(intensities[i]),
                "norm_intensity": float(norm_intensities[i]),
                "rank": int(rank),
            }
            for rank, i in enumerate(sorted_idx, start=1)
        ]

        # 4️⃣ Save ranked summary
        json_path = output_dir / "metric_intensity_summary.json"
        with open(json_path, "w") as f:
            f.write(dumps_safe(ranked_metrics, indent=2))
        log.debug(f"[PhosAnalyzer] Saved metric intensity summary → {json_path}")

        # 5️⃣ Plot top metrics
        top_k = 20
        plt.figure(figsize=(10, 3))
        plt.bar(
            [r["metric"] for r in ranked_metrics[:top_k]],
            [r["mean_intensity"] for r in ranked_metrics[:top_k]],
            color="crimson",
        )
        plt.xticks(rotation=90, fontsize=8)
        plt.title("Top surviving metrics by differential intensity")
        plt.tight_layout()
        plt.savefig(output_dir / "metric_intensity_plot.png", dpi=200)
        plt.close()


        def extract_top_intensity_indices(diff_matrix, k: int = 5) -> list[int]:
            """
            Return indices of the top-K rows by mean absolute intensity.

            Robust to both NumPy arrays and Python lists.
            Handles empty, 1D, or malformed inputs gracefully.
            """
            # ✅ Ensure NumPy array
            if diff_matrix is None:
                return []
            if not isinstance(diff_matrix, np.ndarray):
                try:
                    diff_matrix = np.asarray(diff_matrix, dtype=np.float32)
                except Exception:
                    return []

            # ✅ Defensive: ensure 2D
            if diff_matrix.size == 0:
                return []
            if diff_matrix.ndim < 2:
                diff_matrix = np.expand_dims(diff_matrix, axis=0)

            # ✅ Compute mean absolute intensity per row
            intensities = np.mean(np.abs(diff_matrix), axis=1)

            # ✅ Handle fewer rows than K
            k = min(k, len(intensities))
            if k == 0:
                return []

            # ✅ Sort descending by intensity
            top_idx = np.argsort(intensities)[::-1][:k]

            # ✅ Return as plain list of ints
            return top_idx.tolist()

        # --- 🔦 NEW: capture top-intensity rows
        top_k = 5
        top_idx = extract_top_intensity_indices(diff_matrix, k=top_k)

        # Ensure it's a NumPy array
        if not isinstance(diff_matrix, np.ndarray):
            diff_matrix = np.asarray(diff_matrix, dtype=np.float32)

        # Defensive: ensure 2D shape
        if diff_matrix.ndim < 2:
            diff_matrix = np.expand_dims(diff_matrix, axis=0)

        # Use fancy indexing safely
        top_rows = diff_matrix[top_idx, :].tolist() if len(top_idx) > 0 else []


        # Optional: quick-dump to JSON for inspection
        top_path = output_dir / "top_intensity_rows.json"
        # handle both numpy arrays and Python lists safely

        with open(top_path, "w") as f:
            if hasattr(top_idx, "tolist"):
                top_idx_list = top_idx.tolist()
            else:
                top_idx_list = list(top_idx)
            json.dump({"top_indices": top_idx_list, "top_rows": top_rows}, f, indent=2)
        log.debug(f"[PhosAnalyzer] Extracted top-{top_k} intensity rows → {top_path}")

        return {
            "ranked_metrics": ranked_metrics,
            "top_indices": top_idx,
            "top_rows": top_rows,
        }


    def render_intermodel_delta(
        self,
        A: np.ndarray,                      # HRM matrix (rows=turns, cols=metrics)
        B: np.ndarray,                      # Tiny matrix
        *,
        names_A: Optional[List[str]] = None,
        names_B: Optional[List[str]] = None,
        output_dir: str,
        pos_label: str = "HRM",
        neg_label: str = "Tiny",
        align: str = "canonical",           # "canonical" or "direct"
        alpha: float = 0.97,
        Kc: int = 40,
        Kr: int = 100,
        cmap: str = "seismic",
        metric_names: list[str] | None = None,
        datestamp: bool = True,
    ) -> dict:
        """
        Produce side-by-side grid + Δ (pos−neg) heatmap + JSON summary.
        Returns a meta dict with file paths and intensity rankings.
        """
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(output_dir, f"intermodel_delta_{ts}")

        # 1) Align
        # if align == "canonical":
        #     Y_pos, Y_neg, align_meta = self._canonical_align(A, B, alpha=alpha, Kc=Kc, Kr=Kr)
        # else:
        #     Y_pos, Y_neg = self._direct_align(A, B)
        #     align_meta = {"metric_weights": None, "canonical_layout": None}

        YA, YB, align_meta = self._canonical_align(
                A, B,
                names_A=names_A, names_B=names_B,
                alpha=alpha, Kc=Kc, Kr=Kr
            )

        # 2) Δ
        D = YA - YB
        delta_mass = float(np.mean(D))  # simple summary; keep consistent with other metrics
        overlap = float(np.sum(np.minimum(YA, YB)) / (np.sum(np.maximum(YA, YB)) + 1e-8))

        # 3) Grid figure (Raw/Optimized naming aligned with your EF API style)
        def _save_grid(images, titles, path):
            fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5))
            for ax, img, title in zip(axes, images, titles):
                im = ax.imshow(img, cmap=cmap, aspect="auto")
                ax.set_title(title, fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close(fig)

        comp_path = base + "_comparison.png"
        _save_grid(
            [YA, YB, D],
            [f"Aligned {pos_label}", f"Aligned {neg_label}", f"Δ Field ({pos_label} − {neg_label})"],
            comp_path
        )

        # 4) Standalone Δ heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(D, cmap=cmap, aspect="auto")
        ax.set_title(f"Inter-Model Δ Field — {pos_label} − {neg_label} | ΔMass={delta_mass:.4f}, Overlap={overlap:.4f}")
        fig.colorbar(im, ax=ax, label="Δ intensity")
        plt.tight_layout()
        delta_png = base + "_delta.png"
        plt.savefig(delta_png, dpi=170)
        plt.close(fig)

        # 5) Intensity ranking
        ranks = self._rank_intensity(D, metric_names=metric_names, k_rows=20, k_cols=20)

        # 6) Write JSON
        meta = {
            "timestamp": ts,
            "align": align,
            "pos_label": pos_label,
            "neg_label": neg_label,
            "delta_mass": delta_mass,
            "overlap": overlap,
            "comparison_png": comp_path,
            "delta_png": delta_png,
            "metric_names": metric_names or [],
            "intensity": ranks,
            "alignment": align_meta,
            "shape_pos": list(YA.shape),
            "shape_neg": list(YB.shape),
            "shape_delta": list(D.shape),
            "common_metric_names": align_meta["common_metric_names"],
            "excluded_in_A": [n for n in (names_A or []) if n not in align_meta["common_metric_names"]],
            "excluded_in_B": [n for n in (names_B or []) if n not in align_meta["common_metric_names"]],
        }
        meta_path = base + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(dumps_safe(meta, indent=2))


        return meta


    def _canonical_align(
        self,
        A: np.ndarray,
        B: np.ndarray,
        *,
        names_A: Optional[List[str]] = None,
        names_B: Optional[List[str]] = None,
        alpha: float = 0.97,
        Kc: int = 40,
        Kr: int = 100,
    ):
        """Align A and B into a common feature space safely.

        - If metric names are provided, restrict to their intersection (preserving A order).
        - Otherwise, truncate both to min(n_cols).
        - Then learn spatial layout on A and project both with same weights.
        """
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be 2D matrices")

        nA, dA = A.shape
        nB, dB = B.shape

        # 1) choose common columns
        if names_A and names_B:
            name_to_idx_B = {n: i for i, n in enumerate(names_B)}
            common_names = [n for n in names_A if n in name_to_idx_B]
            if not common_names:
                # fallback: truncate to min width
                d = min(dA, dB)
                A2, B2 = A[:, :d], B[:, :d]
                common_names = [f"metric_{i}" for i in range(d)]
            else:
                idx_A = [i for i, n in enumerate(names_A) if n in name_to_idx_B]
                idx_B = [name_to_idx_B[n] for n in common_names]
                A2 = A[:, idx_A]
                B2 = B[:, idx_B]
        else:
            d = min(dA, dB)
            A2, B2 = A[:, :d], B[:, :d]
            common_names = [f"metric_{i}" for i in range(d)]

        # 2) optimize on A, reuse for B
        from zeromodel.tools.spatial_optimizer import SpatialOptimizer
        opt = SpatialOptimizer(Kc=Kc, Kr=Kr, alpha=alpha)
        opt.apply_optimization([A2])
        w = opt.metric_weights

        YA, _, _ = opt.phi_transform(A2, w, w)
        YB, _, _ = opt.phi_transform(B2, w, w)

        # normalize (defensive)
        def _norm(m):
            m = np.nan_to_num(m)
            maxv = np.max(np.abs(m)) + 1e-8
            return m / maxv

        return _norm(YA), _norm(YB), {
            "common_metric_names": common_names,
            "w": w,
            "width": len(common_names),
            "shape_A": A2.shape,
            "shape_B": B2.shape,
        }

    # --- NEW: direct (no optimizer) alignment helper ----------------------
    def _direct_align(self, A: np.ndarray, B: np.ndarray):
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)

        def _norm01(M):
            M = np.nan_to_num(M.astype(np.float32))
            rng = M.max() - M.min()
            return (M - M.min()) / (rng + 1e-8)

        A = _norm01(A)
        B = _norm01(B)

        r = min(A.shape[0], B.shape[0])
        c = min(A.shape[1], B.shape[1])
        return A[:r, :c], B[:r, :c]

    # --- NEW: intensity ranking helper -----------------------------------
    def _rank_intensity(self, D: np.ndarray, *, metric_names: list[str] | None = None, k_rows: int = 20, k_cols: int = 20):
        D = np.asarray(D, dtype=np.float32)
        if D.ndim < 2:  # safety
            D = np.expand_dims(D, 0)

        row_int = np.mean(np.abs(D), axis=1)
        col_int = np.mean(np.abs(D), axis=0)

        r_idx = np.argsort(row_int)[::-1][:k_rows].tolist()
        c_idx = np.argsort(col_int)[::-1][:k_cols].tolist()

        cols_named = [
            (metric_names[i] if metric_names and i < len(metric_names) else f"metric_{i}")
            for i in c_idx
        ]

        return {
            "top_rows": r_idx,
            "top_cols": c_idx,
            "top_cols_named": cols_named,
            "row_intensities": row_int.tolist(),
            "col_intensities": col_int.tolist(),
        }

    # --- inside ZeroModelService -------------------------------------------------

    def score_vpm_image(
        self,
        vpm_img: np.ndarray,
        *,
        dims: list[str],
        weights: dict[str, float] | None = None,
        order: list[str] | None = None,
        order_decay: float = 0.92,
    ) -> dict:
        """
        Central VPM scorer (service side). Computes simple, stable metrics directly
        from the VPM image (np.ndarray in [0,1] or [0,255]). Returns:
            {
            "scores": {dim: float, ...},
            "overall": float,
            "meta": {...}
            }
        """
        if vpm_img is None:
            return {"scores": {}, "overall": 0.5, "meta": {"reason": "no_image"}}

        img = np.asarray(vpm_img)
        if img.ndim == 3 and img.shape[-1] in (1,3):
            img = img[..., 0] if img.shape[-1] == 1 else np.mean(img, axis=-1)
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        h, w = img.shape

        # helpers
        def _robust01(x):
            x = np.asarray(x, dtype=np.float32).ravel()
            if x.size == 0: return 0.0
            lo, hi = np.percentile(x, [5, 95])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return 0.0
            v = (np.mean(x) - lo) / (hi - lo)
            return float(np.clip(v, 0.0, 1.0))

        # gradients (simple structure cues)
        gy, gx = np.gradient(img)
        grad = np.hypot(gx, gy)                           # edge/complexity signal in [0, ~]
        grad_n = grad / (grad.max() + 1e-8)

        # local entropy (clarity proxy; lower entropy → clearer)
        # compute on 8x8 windows (vectorized-ish)
        ws = max(4, min(h, w) // 16)
        if ws % 2 == 1: ws += 1
        if ws < 4: ws = 4
        # downsampled blocks
        bh = max(1, h // ws)
        bw = max(1, w // ws)
        ent_blocks = []
        for i in range(bh):
            for j in range(bw):
                block = img[i*ws:(i+1)*ws, j*ws:(j+1)*ws]
                hist, _ = np.histogram(block, bins=32, range=(0.0, 1.0), density=True)
                p = hist + 1e-9
                p = p / p.sum()
                ent = -np.sum(p * np.log(p))
                ent_blocks.append(ent)
        ent_blocks = np.array(ent_blocks, dtype=np.float32)
        # normalize entropy to [0,1] (low entropy ⇒ clarity high)
        ent_z = (ent_blocks - ent_blocks.min()) / (ent_blocks.max() - ent_blocks.min() + 1e-8)
        clarity = float(1.0 - np.mean(ent_z))

        # coherence: how aligned gradients are (anisotropy of structure tensor)
        Ixx, Iyy, Ixy = (gx*gx).mean(), (gy*gy).mean(), (gx*gy).mean()
        trace = Ixx + Iyy
        det = Ixx*Iyy - Ixy*Ixy
        # eigenvalues of [[Ixx,Ixy],[Ixy,Iyy]]
        tmp = np.sqrt(max(trace*trace - 4*det, 0.0))
        l1 = 0.5*(trace + tmp) + 1e-8
        l2 = 0.5*(trace - tmp) + 1e-8
        anisotropy = float((l1 - l2) / (l1 + l2))        # 0..1
        coherence = anisotropy

        # coverage / sparsity
        coverage = float(img.mean())                     # how much "mass" is present
        sparsity = 1.0 - coverage

        # complexity: average normalized gradient magnitude
        complexity = float(grad_n.mean())

        # contradiction: local sign-change / roughness proxy
        # (for binary-like vpms this approximates “conflicting regions”)
        diff_h = np.abs(np.diff(img, axis=1)).mean()
        diff_v = np.abs(np.diff(img, axis=0)).mean()
        contradiction = float(_robust01([diff_h, diff_v, grad_n.var()]))

        # confidence: strong, consistent activation & clear edges (not noisy)
        confidence = float(np.clip(0.6*coverage + 0.4*(clarity*(1.0 - contradiction)), 0.0, 1.0))

        # novelty: if you want true novelty vs bank, wire a reference here.
        novelty = 0.5

        # alignment: prefer strong directional structure and clarity
        alignment = float(np.clip(0.5*anisotropy + 0.5*clarity, 0.0, 1.0))

        # coherence already computed; include “relevance” if requested (fallback to coverage)
        candidates = {
            "clarity": clarity,
            "novelty": novelty,
            "confidence": confidence,
            "contradiction": contradiction,
            "coherence": coherence,
            "complexity": complexity,
            "alignment": alignment,
            "coverage": coverage,
            "sparsity": sparsity,
            "relevance": coverage,
        }

        scores = {d: float(np.clip(candidates.get(d, 0.5), 0.0, 1.0)) for d in dims}

        # weighted & order-aware overall
        overall = self._aggregate_dimension_scores(scores, weights=weights or {}, order=order or [], decay=order_decay)

        return {
            "scores": scores,
            "overall": overall,
            "meta": {
                "h": int(h), "w": int(w),
                "grad_mean": float(grad_n.mean()),
                "entropy_mean": float(np.mean(ent_blocks)) if ent_blocks.size else None,
                "ws": int(ws),
            }
        }

    def _aggregate_dimension_scores(
        self,
        scores: dict[str, float],
        *,
        weights: dict[str, float],
        order: list[str],
        decay: float = 0.92,
    ) -> float:
        """
        Combine per-dimension scores with:
        - explicit weights per dim (default 1.0)
        - geometric decay by importance order (first is most important)
        """
        if not scores:
            return 0.5
        # geometric order bonus (first gets 1.0, next *decay, etc.)
        order_boost = {dim: (decay**rank) for rank, dim in enumerate(order)} if order else {}
        wsum = 0.0
        ssum = 0.0
        for dim, val in scores.items():
            w = float(weights.get(dim, 1.0)) * float(order_boost.get(dim, 1.0))
            wsum += w
            ssum += w * float(val)
        return float(ssum / (wsum + 1e-8))

def _finite_mask(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return np.isfinite(a)

def _robust_scale_cols(M: np.ndarray, lo_p=1.0, hi_p=99.0) -> np.ndarray:
    """
    Robust per-column min-max to [0,1] using percentiles (ignores non-finite).
    Columns with flat or invalid ranges are zeroed.
    """
    M = np.asarray(M, dtype=np.float32)
    if M.ndim != 2 or M.size == 0:
        return np.nan_to_num(M)

    X = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0).copy()
    n, d = X.shape
    for j in range(d):
        col = X[:, j]
        # compute percentiles on finite values only
        mask = _finite_mask(col)
        if not mask.any():
            X[:, j] = 0.0
            continue
        v = col[mask]
        lo = np.percentile(v, lo_p)
        hi = np.percentile(v, hi_p)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            X[:, j] = 0.0
            continue
        X[:, j] = np.clip((col - lo) / (hi - lo), 0.0, 1.0)
    return X
