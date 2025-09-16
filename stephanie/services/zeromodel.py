# stephanie/services/zero_model_service.py
from __future__ import annotations

import time
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
    Delegates all rendering to ZeroModel.
    """

    def __init__(self, cfg: Dict[str, Any], memory, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self._initialized = False
        self._pipeline: Optional[PipelineExecutor] = None
        self._gif_fps: int = int(self.cfg.get("zero_model", {}).get("fps", 6))
        self._max_frames: int = int(self.cfg.get("zero_model", {}).get("max_frames", 512))

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

    # --- Public API: arrays in → files out ---

    def render_timeline_from_matrix(
        self,
        matrix: np.ndarray,                # shape (steps, metrics) or (steps, sections, metrics)
        out_path: str,                     # e.g. "reports/vpm/timeline.gif"
        *,
        fps: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a VPM trajectory to ZeroModel; get back a GIF path + a few stats.
        Stephanie does not draw anything.
        """
        assert self._pipeline is not None, "ZeroModelAdapter not initialized"
        ctx: Dict[str, Any] = options.copy() if options else {}

        # Declare the VPM to the pipeline; it will render frames around stages.
        ctx["vpm"] = matrix
        if metrics:
            ctx["metric_names"] = metrics  # used by normalize stage if needed

        # Tell the pipeline to record frames.
        gif = GifLogger(fps=(fps or self._gif_fps), max_frames=self._max_frames)
        ctx["gif_logger"] = gif

        # Run the pipeline; it will call gif.add_frame(...) pre/post each stage.
        self._pipeline.process(ctx)

        # Save the animation (ZeroModel’s logger owns the pixels)
        gif.save_gif(out_path, fps=(fps or self._gif_fps))
        return {
            "output_path": out_path,
            "frames": len(gif.frames),
            "shape": list(matrix.shape),
        }

    def generate_summary_vpm_tiles(
        self,
        *,
        vpm_data: Dict[str, Any],  # {"doc_id", "title", "metrics": {"A","B","C"}, "iterations": [...]}
        output_dir: str,
        filenames: Tuple[str, str] = ("quality.png", "iteration.png"),
    ) -> Dict[str, Any]:
        """
        Make 2 static tiles (quality + iteration trace). Implementation delegates to ZeroModel’s
        GIF logger to compose the panel, then writes PNGs via the logger’s new `save_png(...)`
        (see patch below). Stephanie never touches PIL.
        """
        assert self._pipeline is not None, "ZeroModelAdapter not initialized"

        # 1) Build tiny VPM rows for the “quality tile” (A, B, C columns)
        #    We encode one “step” that has 3 columns so the logger can draw a composite panel.
        metrics = ["overall", "coverage", "faithfulness", "structure", "no_halluc"]
        abc = []
        for k in ("A", "B", "C"):
            m = (vpm_data.get("metrics", {}).get(k) or {})
            abc.append([float(m.get(x, 0.0)) for x in metrics])
        # (step=0, cols=3, metrics=5) → the preview encoder will handle either shape.
        mat_quality = np.asarray([abc], dtype=np.float32).squeeze(0)  # (cols=3, metrics)

        # 2) Iteration trace: 1 column over T steps = overall score over time
        iters = vpm_data.get("iterations", []) or []
        track = np.asarray([[float(x.get("best_candidate_score", 0.0))] for x in iters], dtype=np.float32)
        mat_iter = track  # (steps, 1)

        gif = GifLogger()
        # Quality panel (single frame)
        gif.add_frame(mat_quality, metrics={"step": 0})
        # Iteration panel (single frame)
        gif.add_frame(mat_iter, metrics={"step": 1})

        # Save two PNGs (requires small helper in ZeroModel; see patch below)
        quality_path = f"{output_dir.rstrip('/')}/{filenames[0]}"
        iter_path    = f"{output_dir.rstrip('/')}/{filenames[1]}"
        # gif.save_gif(path=quality_path, fps=6, optimize=True, loop=0)
        # gif.save_gif(path=iter_path, fps=6, OK so optimize=True, loop=0)

        return {"quality_tile_path": quality_path, "iteration_trace_path": iter_path}


    def emit_iteration_tile(
        self,
        *,
        doc_id: str,
        iteration: int,
        metrics: Dict[str, float],
        output_dir: str = "reports/vpm/iters",
    ) -> str:
        """
        Emit a single PNG (or 1-frame GIF fallback) visualizing an iteration score triple.
        Metrics accepted: overall, knowledge_verification, hrm_score (others ignored).
        """
        assert self._pipeline is not None, "ZeroModelAdapter not initialized"
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 1 row x N metrics matrix
        keys = ["overall", "knowledge_verification", "hrm_score"]
        row = [float(metrics.get(k, 0.0) or 0.0) for k in keys]
        mat = np.asarray([row], dtype=np.float32)

        # Use logger to own pixels
        gif = GifLogger(fps=1, max_frames=4)
        gif.add_frame(mat, metrics={"iter": iteration})

        out_png = os.path.join(output_dir, f"{doc_id}_iter_{iteration:02d}.png")
        if hasattr(gif, "save_png"):
            gif.save_png(out_png, frame_index=-1)
        else:
            # Fallback: 1-frame GIF if PNG support isn’t available yet
            out_png = os.path.join(output_dir, f"{doc_id}_iter_{iteration:02d}.gif")
            gif.save_gif(out_png, fps=1)
        return out_png

    def _emit_timeline(self, mat: np.ndarray, out_path: str, fps: Optional[int] = None) -> Dict[str, Any]:
        """Small wrapper so older callers work."""
        return self.render_timeline_from_matrix(mat, out_path, fps=fps or self._gif_fps)

    def generate_summary_vpm_tiles(
        self,
        *,
        vpm_data: Dict[str, Any],
        output_dir: str,
        filenames: Tuple[str, str] = ("quality.png", "iteration.png"),
    ) -> Dict[str, Any]:
        """
        Now actually writes files. Uses GifLogger.save_png if present; else 1-frame GIFs.
        """
        assert self._pipeline is not None, "ZeroModelAdapter not initialized"
        import os
        os.makedirs(output_dir, exist_ok=True)

        metrics = ["overall", "coverage", "faithfulness", "structure", "no_halluc"]
        abc = []
        for k in ("A", "B", "C"):
            m = (vpm_data.get("metrics", {}).get(k) or {})
            abc.append([float(m.get(x, 0.0)) for x in metrics])
        mat_quality = np.asarray(abc, dtype=np.float32)               # (3,5)

        iters = vpm_data.get("iterations", []) or []
        track = np.asarray([[float(x.get("best_candidate_score", 0.0))] for x in iters], dtype=np.float32)
        mat_iter = track if track.size else np.asarray([[0.0]], dtype=np.float32)   # (T,1) or (1,1)

        gif = GifLogger(fps=1, max_frames=self._max_frames)
        # encode two frames so downstream can show a mini-gallery if desired
        gif.add_frame(mat_quality, metrics={"panel": "quality"})
        gif.add_frame(mat_iter,    metrics={"panel": "iteration"})

        quality_path = os.path.join(output_dir, filenames[0])
        iter_path    = os.path.join(output_dir, filenames[1])

        if hasattr(gif, "save_png"):
            gif.save_png(quality_path, frame_index=0)
            gif.save_png(iter_path,    frame_index=1)
        else:
            # fallback: 1-frame GIFs
            gif_q = GifLogger(fps=1) 
            gif_q.add_frame(mat_quality, metrics={})
            gif_q.save_gif(quality_path.replace(".png", ".gif"), fps=1)
            gif_i = GifLogger(fps=1) 
            gif_i.add_frame(mat_iter,    metrics={})
            gif_i.save_gif(iter_path.replace(".png", ".gif"), fps=1)

        return {"quality_tile_path": quality_path, "iteration_trace_path": iter_path}
