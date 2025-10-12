# stephanie/services/zero_model_service.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from stephanie.utils.json_sanitize import dumps_safe
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
from pathlib import Path

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
        _logger.debug("ZeroModelService initialized")

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
        _logger.debug("ZeroModelService shutdown")

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
        _logger.debug(f"Timeline opened for run_id={run_id}")

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
            sess = self._sessions.get(run_id)

        # Normalize
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

        # 2️⃣ Initialize metrics_order if missing or dummy
        if not sess.metrics_order or sess.metrics_order == []:
            sess.metrics_order = list(metrics_columns)
            _logger.debug(f"[ZeroModelService] Metric order initialized → {sess.metrics_order}")
        if len(metrics_columns) != len(sess.metrics_order):
            _logger.warning(
                """[ZeroModelService] Mismatched metrics length for run_id=%s: 
                expected %d but got %d""",
                run_id,
                len(sess.metrics_order),
                len(metrics_columns)
            )
            # Find the specific differences
            expected_set = set(sess.metrics_order)
            received_set = set(metrics_columns)
            missing_columns = expected_set - received_set
            extra_columns = received_set - expected_set
            # a lot of work but we need to understand why the metrics are out of shape
            _logger.warning(
                """[ZeroModelService] Mismatched metrics length for run_id=%s: 
                expected %d but got %d
                Missing columns: %s
                Extra columns: %s
                Expected: %s
                Received: %s""",
                run_id,
                len(sess.metrics_order),
                len(metrics_columns),
                list(missing_columns) if missing_columns else "None",
                list(extra_columns) if extra_columns else "None",
                sess.metrics_order,
                metrics_columns
            )

        # ✅ Expand gracefully if new metrics appear later
        if len(metrics_columns) > len(sess.metrics_order):
            for name in metrics_columns:
                if name not in sess.metrics_order:
                    sess.metrics_order.append(name)
            # pad older rows
            for i in range(len(sess.rows)):
                sess.rows[i] += [0.0] * (len(sess.metrics_order) - len(sess.rows[i]))

        # Map columns to aligned row
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
        if mat.shape[0] < 3 or mat.shape[1] < 2:
            _logger.warning(f"[ZeroModelService] Too few rows ({mat.shape}) for run_id={run_id}, deferring finalize.")
            # Keep session alive for a few more seconds
            self._sessions[run_id] = sess
            await asyncio.sleep(4.0)
            mat = sess.as_matrix()

        if mat.size == 0 or mat.shape[0] < 2:
            _logger.warning(f"[ZeroModelService] Empty or too small matrix for run_id={run_id}, skipping render.")
            return {"status": "empty", "matrix": mat}

        _logger.debug(
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
        _logger.debug(f"ZeroModelService: summary image saved → {summary_path}")


        # ------------------------------------------------------------------
        # 🌌 Auto-generate epistemic field (optional)
        # ------------------------------------------------------------------
        try:
            # Build small contrastive populations from the current matrix
            # Here we treat the first half as "positive" and second half as "negative"
            if mat.size >= 4 and mat.shape[0] > 2:
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
                _logger.debug(
                    f"[ZeroModelService] 🧠 Epistemic field auto-generated "
                    f"(ΔMass={field_meta['delta_mass']:.4f}) → {field_meta['png']}"
                )
            else:
                _logger.warning("[ZeroModelService] Matrix too small for epistemic field generation.")
        except Exception as e:
            _logger.warning("[ZeroModelService] Epistemic field generation failed: %s", e)


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

        return {
            "status": "ok",
            "matrix": mat,
            "metric_names": sess.metrics_order,
            **res,
            "meta_path": meta_path,
            "summary_path": summary_path,
        }

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
        _logger.debug(f"ZeroModelService: rendered {len(gif.frames)} frames → {out_path}")

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

            _logger.debug(f"ZeroModelService: static VPM summary saved → {png_path}")
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
            _logger.debug(
                f"Matrix sorted by first index "
                f"(min={col0.min():.4f}, max={col0.max():.4f}, rows={matrix.shape[0]})"
            )
            return sorted_matrix

        except Exception as e:
            _logger.warning("[ZeroModelService] sort_on_first_index failed: %s", e)
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
        aggregate: bool = False,
        metric_names: Optional[List[str]] = None,
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
            _logger.warning("[ZeroModelService] Skipping epistemic field generation — insufficient data.")
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
        opt = SpatialOptimizer(Kc=Kc, Kr=Kr, alpha=alpha)
        opt.apply_optimization([X_pos])
        w = opt.metric_weights
        layout = opt.canonical_layout
        
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

        _logger.debug(f"Epistemic field generated: +mass={mass_pos:.4f}, -mass={mass_neg:.4f}, Δ={delta_mass:.4f}, overlap={overlap:.4f}")

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
            gif_logger = GifLogger(max_frames=100)

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
                gif_logger.add_frame(frame, metrics={"stage": title})
                plt.close(fig)

            gif_logger.save_gif(gif_path, fps=1)
            _logger.debug(f"Transformation GIF saved → {gif_path}")

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

        _logger.debug(f"[ZeroModelService] Metric reordering preserved {len(reordered_metric_names)} names")

        # ================================================================
        # 8️⃣ Subfield Extraction — Q-field, Energy-field, Overlay
        # ================================================================
        try:
            # Helper normalization
            def _normalize(mat):
                mat = np.nan_to_num(mat)
                return mat / (np.max(np.abs(mat)) + 1e-8) if mat.size else mat

            # Helper to extract column subset by keyword
            def _subset_field(names: List[str], keywords: List[str]) -> np.ndarray:
                if not names:
                    return np.zeros_like(diff)
                cols = [i for i, n in enumerate(names)
                        if any(k in n.lower() for k in keywords)]
                if not cols:
                    return np.zeros_like(diff)
                return diff[:, cols]

            # --- Extract Q-value and Energy subfields ---
            q_field = _subset_field(reordered_metric_names, ["q_value"])
            e_field = _subset_field(reordered_metric_names, ["energy"])

            q_field_norm = _normalize(q_field)
            e_field_norm = _normalize(e_field)

            # --- Compute overlay ---
            overlay = q_field_norm - e_field_norm

            # --- Visualization helper ---
            def _save_field_image(mat, title, suffix):
                if mat.size == 0:
                    return None
                fig, ax = plt.subplots(figsize=(8, 5))
                im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
                ax.set_title(title, fontsize=10)
                fig.colorbar(im, ax=ax, label="Δ Intensity")
                plt.tight_layout()
                out_path = f"{base}_{suffix}.png"
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                return out_path

            q_path = _save_field_image(q_field_norm, "Q-Field (Value Surface)", "q_field")
            e_path = _save_field_image(e_field_norm, "Energy Field (Uncertainty Surface)", "energy_field")
            o_path = _save_field_image(overlay, "Q–Energy Overlay (Value−Uncertainty)", "overlay")

            # --- Simple quantitative correlation between Q and Energy ---
            corr = float(np.corrcoef(
                q_field_norm.flatten(), e_field_norm.flatten()
            )[0, 1]) if q_field_norm.size and e_field_norm.size else None

            _logger.debug(
                f"[ZeroModelService] Subfields saved → Q:{bool(q_path)} | E:{bool(e_path)} | "
                f"Overlay:{bool(o_path)} | Corr(Q,E)={corr}"
            )

        except Exception as e:
            _logger.warning("[ZeroModelService] Subfield extraction failed: %s", e)


        # ------------------------------- 
        # Prepare visual stages
        # -------------------------------
        raw_good = _normalize_field(X_pos)
        raw_bad  = _normalize_field(X_neg)
        opt_good = Y_pos
        opt_bad  = Y_neg
        combined = diff

        titles = [
            "Raw Good (Before Optimization)",
            "Optimized Good (Spatial Calculus)",
            "Raw Bad (Before Optimization)",
            "Optimized Bad (Spatial Calculus)",
            "Differential Field (Good − Bad)",
        ]
        images = [raw_good, opt_good, raw_bad, opt_bad, combined]

        # Render comparison grid + GIF
        comparison_path = _make_visual_grid(images, titles, base)
        transition_gif_path = base + "_transform.gif"
        _make_transition_gif(images, titles, transition_gif_path)
        _logger.debug(f"Epistemic field visual comparison saved → {comparison_path}")


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
            "png": base + ".png",
            "gif": gif_path,
            "diff_matrix": diff.tolist(),
            "metric_names": reordered_metric_names,
        }

        meta_path = base + ".json"
        text=dumps_safe(meta)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(text)

        _logger.debug(f"Epistemic field saved → {meta_path}")
        return meta


    def analyze_differential_field(self, diff_matrix: np.ndarray, metric_names: list[str], output_dir: str):
        """
        Analyze the differential field (Good - Bad) to identify surviving high-intensity metrics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if diff_matrix is None:
            _logger.warning("[PhosAnalyzer] Empty diff_matrix — skipping analysis.")
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
            _logger.warning("[PhosAnalyzer] Zero-intensity field — skipping plot.")
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
            json.dump(ranked_metrics, f, indent=2)
        _logger.debug(f"[PhosAnalyzer] Saved metric intensity summary → {json_path}")

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
        _logger.debug(f"[PhosAnalyzer] Extracted top-{top_k} intensity rows → {top_path}")

        return {
            "ranked_metrics": ranked_metrics,
            "top_indices": top_idx,
            "top_rows": top_rows,
        }


