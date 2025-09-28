# stephanie/services/zero_model_service.py
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.tools.gif_logger import GifLogger  # NOTE: capital G only on Gif

from stephanie.services.service_protocol import Service

_DEFAULT_PIPELINE = [
    {"name": "normalize", "params": {}},            # infers metric names if not given
    {"name": "feature_engineering", "params": {}},
    {"name": "organization", "params": {"strategy": "spatial"}},
    # optional: {"name": "occlusion_explainer", "params": {"window_size": 5}},
]

class ZeroModelService(Service):
    """
    Thin adapter so Stephanie never touches image libs.
    Delegates all rendering to ZeroModel via its pipeline + GifLogger.
    """

    def __init__(self, cfg: Dict[str, Any], memory, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self._initialized = False
        self._pipeline: Optional[PipelineExecutor] = None
        zm_cfg = self.cfg.get("zero_model", {}) or {}
        self._gif_fps: int = int(zm_cfg.get("fps", 6))
        self._max_frames: int = int(zm_cfg.get("max_frames", 512))

    # --- Service Protocol ---
    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return
        pipeline_cfg = self.cfg.get("zero_model", {}).get("pipeline") or _DEFAULT_PIPELINE
        self._pipeline = PipelineExecutor(pipeline_cfg)
        self._initialized = True

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {"fps": self._gif_fps, "max_frames": self._max_frames},
            "dependencies": {"zeromodel": "ok" if self._pipeline else "missing"},
        }

    def shutdown(self) -> None:
        self._pipeline = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "zeromodel-service-v1"

    # --------------------------
    # Helpers
    # --------------------------
    def _stamp(self, path: str, enable: bool = False) -> str:
        """Append UTC timestamp to filename if enabled."""
        if not enable:
            return path
        root, ext = os.path.splitext(path)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{root}_{ts}{ext}"

    # --------------------------
    # Public API: arrays in → files out
    # --------------------------
    def render_timeline_from_matrix(
        self,
        matrix: np.ndarray,                # shape: (steps, metrics) or (steps, sections, metrics)
        out_path: str,                     # e.g. "reports/vpm/timeline.gif"
        *,
        fps: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        datestamp: bool = False,
    ) -> Dict[str, Any]:
        """
        Send a VPM trajectory to ZeroModel; get back a GIF path + a few stats.
        Stephanie does not draw anything.
        """
        assert self._pipeline is not None, "ZeroModelService not initialized"

        ctx: Dict[str, Any] = (options or {}).copy()
        ctx["vpm"] = matrix
        if metrics:
            ctx["metric_names"] = metrics  # used by normalize stage if needed

        # Let GifLogger own frames; set fps only on save, not on init.
        gif = GifLogger(max_frames=self._max_frames)
        ctx["gif_logger"] = gif

        # Run the pipeline; it will call gif.add_frame(...) pre/post each stage.
        self._pipeline.process(ctx)

        # Save animation (ZeroModel’s logger owns the pixels)
        out_path = self._stamp(out_path, datestamp)
        gif.save_gif(out_path, fps=(fps or self._gif_fps))
        return {
            "output_path": out_path,
            "frames": len(gif.frames),
            "shape": list(matrix.shape),
        }

    def generate_summary_vpm_tiles(
        self,
        *,
        vpm_data: Dict[str, Any],               # {"doc_id","title","metrics":{"A","B","C"},"iterations":[...]}
        output_dir: str,
        filenames: Tuple[str, str] = ("quality.png", "iteration.png"),
        datestamp: bool = True,
    ) -> Dict[str, Any]:
        """
        Build two static tiles via GifLogger: 'quality' (A/B/C panel) and 'iteration' (score over time).
        Uses save_png if available; otherwise falls back to 1-frame GIFs.
        """
        assert self._pipeline is not None, "ZeroModelService not initialized"
        os.makedirs(output_dir, exist_ok=True)

        # Quality panel: rows A/B/C × metrics
        metric_names = ["overall", "coverage", "faithfulness", "structure", "no_halluc"]
        abc = []
        for k in ("A", "B", "C"):
            m = (vpm_data.get("metrics", {}).get(k) or {})
            abc.append([float(m.get(x, 0.0)) for x in metric_names])
        mat_quality = np.asarray(abc, dtype=np.float32)  # (3, 5)

        # Iteration panel: 1 column across T steps (overall score trajectory)
        iters = vpm_data.get("iterations", []) or []
        track = np.asarray([[float(x.get("best_candidate_score", 0.0))] for x in iters], dtype=np.float32)
        mat_iter = track if track.size else np.asarray([[0.0]], dtype=np.float32)  # (T,1) or (1,1)

        gif = GifLogger(max_frames=self._max_frames)
        gif.add_frame(mat_quality, metrics={"panel": "quality"})
        gif.add_frame(mat_iter,    metrics={"panel": "iteration"})

        q_path = self._stamp(os.path.join(output_dir, filenames[0]), datestamp)
        i_path = self._stamp(os.path.join(output_dir, filenames[1]), datestamp)

        if hasattr(gif, "save_png"):
            gif.save_png(q_path, frame_index=0)
            gif.save_png(i_path, frame_index=1)
        else:
            # fallback: 1-frame GIFs
            gif_q = GifLogger(max_frames=4)
            gif_q.add_frame(mat_quality, metrics={})
            gif_q.save_gif(q_path[:-4] + ".gif", fps=1)
            gif_i = GifLogger(max_frames=4)
            gif_i.add_frame(mat_iter, metrics={})
            gif_i.save_gif(i_path[:-4] + ".gif", fps=1)

        return {"quality_tile_path": q_path, "iteration_trace_path": i_path}

    def emit_iteration_tile(
        self,
        *,
        doc_id: str,
        iteration: int,
        metrics: Dict[str, float],
        output_dir: str = "reports/vpm/iters",
        datestamp: bool = True,
    ) -> str:
        """
        Emit a single static tile (PNG if supported, else 1-frame GIF) for one iteration.
        Accepts keys: overall, knowledge_verification, hrm_score (others ignored).
        """
        assert self._pipeline is not None, "ZeroModelService not initialized"
        os.makedirs(output_dir, exist_ok=True)

        keys = ["overall", "knowledge_verification", "hrm_score"]
        row = [float(metrics.get(k, 0.0) or 0.0) for k in keys]
        mat = np.asarray([row], dtype=np.float32)  # (1, 3)

        gif = GifLogger(max_frames=4)
        gif.add_frame(mat, metrics={"iter": iteration})

        base_png = os.path.join(output_dir, f"{doc_id}_iter_{iteration:02d}.png")
        out_png = self._stamp(base_png, datestamp)

        if hasattr(gif, "save_png"):
            gif.save_png(out_png, frame_index=-1)
        else:
            out_gif = out_png[:-4] + ".gif"
            gif.save_gif(out_gif, fps=1)
            out_png = out_gif
        return out_png

    # Back-compat helper for any older callers
    def _emit_timeline(self, mat: np.ndarray, out_path: str, fps: Optional[int] = None) -> Dict[str, Any]:
        return self.render_timeline_from_matrix(mat, out_path, fps=(fps or self._gif_fps))
