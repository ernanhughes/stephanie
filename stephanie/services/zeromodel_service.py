# stephanie/services/zero_model_service.py
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stephanie.services.event_service import EventService
from stephanie.services.service_protocol import Service
from zeromodel.tools.spatial_optimizer import SpatialOptimizer
from zeromodel.tools.gif_logger import GifLogger
from zeromodel.pipeline.executor import PipelineExecutor

import matplotlib
if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

_logger = logging.getLogger(__name__)

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
        """Return a numpy matrix for all stored rows."""
        if not self.rows:
            return np.zeros((0, len(self.metrics_order)), dtype=np.float32)
        return np.asarray(self.rows, dtype=np.float32)


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
        _logger.info("ZeroModelService initialized")

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
        _logger.info("ZeroModelService shutdown")

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
        order = metrics or [
            "metric", "value", "visits", "bug",
            "action_draft", "action_improve", "action_debug",
        ]
        odir = out_dir or self._out_dir
        os.makedirs(odir, exist_ok=True)
        self._sessions[run_id] = _TimelineSession(run_id=run_id, metrics_order=order, out_dir=odir)
        _logger.info(f"Timeline opened for run_id={run_id}")

    def timeline_append_row(
        self,
        run_id: str,
        *,
        metrics_columns: List[str],
        metrics_values: List[Any],
    ) -> None:
        """
        Append a single prompt's metrics vector to the session timeline.

        - Each row corresponds to metrics_columns (the same order every time).
        - Values are normalized per scalar:
            0–1 stays as-is, 1–100 → /100, >100 → 1.0, <0 → 0.0.
        - If columns differ or expand, we pad existing rows with zeros.
        - Never fails, never logs errors.
        """
        sess = self._sessions.get(run_id)
        if not sess:
            return

        # 1️⃣ Normalize all incoming values (per your rule)
        normalized = []
        for v in metrics_values:
            try:
                f = float(v)
            except Exception:
                f = 0.0
            if f < 0:
                f = abs(f)
            elif f > 1.0:
                f = min(f / 100.0, 1.0)
            normalized.append(f)

        # 2️⃣ Initialize metrics_order if first row
        if not sess.metrics_order:
            sess.metrics_order = list(metrics_columns)

        # 3️⃣ Handle expanded metric sets gracefully
        if len(metrics_columns) > len(sess.metrics_order):
            for name in metrics_columns:
                if name not in sess.metrics_order:
                    sess.metrics_order.append(name)
            for i in range(len(sess.rows)):
                sess.rows[i] += [0.0] * (len(sess.metrics_order) - len(sess.rows[i]))

        # Align
        name_to_val = dict(zip(metrics_columns, normalized))
        row = [float(name_to_val.get(name, 0.0)) for name in sess.metrics_order]
        sess.rows.append(row)

    # ------------------------------------------------------------------ #
    # FINALIZE & RENDER
    # ------------------------------------------------------------------ #
    async def timeline_finalize(
        self,
        run_id: str,
        *,
        fps: Optional[int] = None,
        datestamp: bool = True,
        out_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        sess = self._sessions.pop(run_id, None)
        if not sess:
            return {"status": "noop", "reason": "no_session"}

        mat = sess.as_matrix()
        _logger.info(
            f"ZeroModelService: finalizing timeline for run_id={run_id} "
            f"with {mat.shape[0]} steps and {mat.shape[1]} metrics"
        )

        # Timestamped output dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_prefix = str(run_id)[:8]
        run_dir = os.path.join(sess.out_dir, f"run_{run_prefix}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        base_name = f"vpm_timeline_{run_prefix}_{timestamp}"
        gif_path = os.path.join(run_dir, base_name + ".gif")

        # Render animated timeline GIF
        res = self.render_timeline_from_matrix(
            mat,
            gif_path,
            fps=(fps or self._gif_fps),
            metrics=sess.metrics_order,
            options={"panel": "timeline"},
            datestamp=datestamp,
        )

        # Write meta JSON
        meta_path = os.path.join(run_dir, base_name + ".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "metrics": sess.metrics_order,
                    "shape": res["shape"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Render static summary PNG
        summary_path = self.render_static_summary(
            mat,
            out_path=run_dir,
            metrics=sess.metrics_order,
            label="timeline",
            timestamp=timestamp,
        )
        _logger.info(f"ZeroModelService: summary image saved → {summary_path}")

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

        return {"status": "ok", **res, "meta_path": meta_path, "summary_path": summary_path}

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
    ) -> Dict[str, Any]:
        assert self._pipeline is not None, "ZeroModelService not initialized"

        if not isinstance(matrix, np.ndarray) or matrix.size == 0:
            ncols = 1 if metrics is None else max(1, len(metrics))
            matrix = np.zeros((1, ncols), dtype=np.float32)

        gif = GifLogger(max_frames=self._max_frames)
        fps = fps or self._gif_fps

        sorted_matrix = self.sort_on_first_index(matrix)
        for i in range(sorted_matrix.shape[0]):
            row = sorted_matrix[i : i + 1, :]
            vpm_out, _ = self._pipeline.run(row, {"enable_gif": False})
            gif.add_frame(
                vpm_out,
                metrics={
                    "step": i,
                    "loss": 1 - row.mean(),
                    "val_loss": row.mean(),
                    "acc": row.std(),
                },
            )

        gif.save_gif(out_path, fps=fps)
        _logger.info(f"ZeroModelService: rendered {len(gif.frames)} frames → {out_path}")

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
            _logger.warning("render_static_summary called with empty matrix")
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

            _logger.info(f"ZeroModelService: static VPM summary saved → {png_path}")
            return png_path
        except Exception as e:
            _logger.error(f"render_static_summary failed: {e}")
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
            _logger.info(
                f"Matrix sorted by first index "
                f"(min={col0.min():.4f}, max={col0.max():.4f}, rows={matrix.shape[0]})"
            )
            return sorted_matrix

        except Exception as e:
            _logger.warning(f"sort_on_first_index failed: {e}")
            return matrix



    def generate_epistemic_field(
        self,    
        pos_matrices: list[np.ndarray],
        neg_matrices: list[np.ndarray],
        output_dir: str = "data/epistemic_field",
        alpha: float = 0.97,
        Kc: int = 40,
        Kr: int = 100,
        fps: int = 8,
        cmap: str = "seismic",
    ) -> dict:
        """
        Generate the ZeroModel Epistemic Field — a contrastive visual intelligence map.

        - Learns canonical spatial layout from positive samples
        - Projects negative samples into same layout
        - Combines them into a single differential VPM
        - Renders static + animated representations
        """
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(output_dir, f"epistemic_field_{ts}")

        # 1️⃣ Stack population matrices
        X_pos = np.vstack(pos_matrices)
        X_neg = np.vstack(neg_matrices)

        # 2️⃣ Learn canonical layout from positives
        opt = SpatialOptimizer(Kc=Kc, Kr=Kr, alpha=alpha)
        opt.apply_optimization([X_pos])
        w = opt.metric_weights
        layout = opt.canonical_layout

        # 3️⃣ Apply canonical transform to both
        Y_pos, _, _ = opt.phi_transform(X_pos, w, w)
        Y_neg, _, _ = opt.phi_transform(X_neg, w, w)

        # 4️⃣ Compute epistemic field (differential)
        diff = Y_pos - Y_neg
        mass_pos = opt.top_left_mass(Y_pos)
        mass_neg = opt.top_left_mass(Y_neg)
        delta_mass = mass_pos - mass_neg

        _logger.info(
            f"Epistemic field generated: +mass={mass_pos:.4f}, -mass={mass_neg:.4f}, Δ={delta_mass:.4f}"
        )

        # 5️⃣ Render static field image
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(diff, cmap=cmap, aspect="auto")
        ax.set_title(f"ZeroModel Epistemic Field — ΔMass={delta_mass:.4f}")
        fig.colorbar(im, ax=ax, label="Δ Intensity (Pos−Neg)")
        plt.tight_layout()
        png_path = base + ".png"
        plt.savefig(png_path, dpi=150)
        plt.close(fig)

        # 6️⃣ Optional animated timeline view
        gif_logger = GifLogger(max_frames=300)
        pipeline = PipelineExecutor([
            {"stage": "normalize", "params": {}},
            {"stage": "organization", "params": {"strategy": "spatial"}},
        ])

        for i in range(diff.shape[0]):
            row = diff[i:i+1, :]
            out, _ = pipeline.run(row, {"enable_gif": False})
            gif_logger.add_frame(out, metrics={"step": i, "mass": float(row.mean())})

        gif_path = base + ".gif"
        gif_logger.save_gif(gif_path, fps=fps)

        # 7️⃣ Persist metadata
        meta = {
            "timestamp": ts,
            "mass_pos": mass_pos,
            "mass_neg": mass_neg,
            "delta_mass": delta_mass,
            "metric_weights": w.tolist(),
            "canonical_layout": layout.tolist(),
            "shape": diff.shape,
            "png": png_path,
            "gif": gif_path,
        }
        meta_path = base + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        _logger.info(f"Epistemic field saved to {output_dir}")
        return meta
