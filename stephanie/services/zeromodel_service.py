# stephanie/services/zero_model_service.py
from __future__ import annotations

import os
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from stephanie.services.service_protocol import Service
from stephanie.services.event_service import EventService
from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.tools.gif_logger import GifLogger  # ZeroModel owns rendering


_logger = logging.getLogger(__name__)

_DEFAULT_PIPELINE = [
    {"stage": "normalize", "params": {}},
    {"stage": "feature_engineering", "params": {}},
    {"stage": "organization", "params": {"strategy": "spatial"}},
]

@dataclass
class _TimelineSession:
    run_id: str
    metrics_order: List[str]
    rows: List[List[float]] = field(default_factory=list)
    meta: List[Dict[str, Any]] = field(default_factory=list)
    out_dir: str = "data/vpms"

    def as_matrix(self) -> np.ndarray:
        if not self.rows:
            return np.zeros((0, len(self.metrics_order)), dtype=np.float32)
        return np.asarray(self.rows, dtype=np.float32)

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
        self._bus_bindings: Dict[str, List[str]] = {}  # run_id -> [subjects]

    # --- Service Protocol ---
    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return
        pipeline_cfg = self.cfg.get("zero_model", {}).get("pipeline") or _DEFAULT_PIPELINE
        self._pipeline = PipelineExecutor(pipeline_cfg)
        self._evt = EventService(self.cfg, self.memory, self.logger)
        self._evt.initialize()
        self._initialized = True

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

    @property
    def name(self) -> str:
        return "zeromodel-service-v2"

    # --------------------------
    # Agent-controlled API (direct)
    # --------------------------
    def timeline_open(self, run_id: str, *, metrics: Optional[List[str]] = None, out_dir: Optional[str] = None) -> None:
        """Create an isolated timeline session for a run."""
        if not run_id:
            raise ValueError("timeline_open requires run_id")
        if run_id in self._sessions:
            return
        order = metrics or ["metric","value","visits","bug","action_draft","action_improve","action_debug"]
        odir = out_dir or self._out_dir
        os.makedirs(odir, exist_ok=True)
        self._sessions[run_id] = _TimelineSession(run_id=run_id, metrics_order=order, out_dir=odir)


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
            old_len = len(sess.metrics_order)
            # extend metric order to include new names
            for name in metrics_columns:
                if name not in sess.metrics_order:
                    sess.metrics_order.append(name)
            # pad previous rows with zeros to match new width
            new_len = len(sess.metrics_order)
            for i in range(len(sess.rows)):
                old = sess.rows[i]
                if len(old) < new_len:
                    sess.rows[i] = old + [0.0] * (new_len - len(old))

        # 4️⃣ Align incoming row to the session order
        name_to_val = dict(zip(metrics_columns, normalized))
        row = [float(name_to_val.get(name, 0.0)) for name in sess.metrics_order]

        # 5️⃣ Append
        sess.rows.append(row)

    async def timeline_finalize(self, run_id: str, *, fps: Optional[int] = None, datestamp: bool = True) -> Dict[str, Any]:
        """Close session, render GIF, emit bus event, and persist companion JSON."""
        sess = self._sessions.pop(run_id, None)
        if not sess:
            return {"status": "noop", "reason": "no_session"}

        mat = sess.as_matrix()  # (steps, metrics)
        _logger.info(f"ZeroModelService: finalizing timeline for run_id={run_id} with {mat.shape[0]} steps and {mat.shape[1]} metrics")

        base = os.path.join(sess.out_dir, f"vpm_timeline_{run_id}")
        # guarantee .gif at the call site
        gif_path = base if base.lower().endswith(".gif") else base + ".gif"

        _logger.info(f"ZeroModelService: rendering timeline for run_id={run_id} to {gif_path} with fps={fps or self._gif_fps}")
        res = self.render_timeline_from_matrix(
            mat,
            gif_path,  # <-- always ends with .gif
            fps=(fps or self._gif_fps),
            metrics=sess.metrics_order,
            options={"panel": "timeline"},
            datestamp=datestamp,
        ) 

        # robust meta path derivation even if a dot is missing for some reason
        op = res["output_path"]
        dot = op.rfind(".")
        meta_path = (op[:dot] if dot != -1 else op) + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "metrics": sess.metrics_order,
                "rows": sess.meta,
                "shape": res["shape"],
            }, f, ensure_ascii=False, indent=2)

        # notify UI via EventService (enveloped, idempotent, DLQ-capable)
        if self._evt:
            await self._evt.publish("arena.ats.timeline_ready", {
                "run_id": run_id,
                "gif": res["output_path"],
                "json": meta_path,
                "steps": mat.shape[0],
                "metrics": sess.metrics_order,
            })

        return {"status": "ok", **res, "meta_path": meta_path}

    # --------------------------
    # Optional: bus-attached mode (agent asks service to listen)
    # --------------------------
    async def timeline_attach_bus(self, run_id: str, *, node_subject: str, report_subject: str) -> None:
        """Attach dynamic subjects to this run; unsubscribes later via detach."""
        self.timeline_open(run_id)
        bindings = self._bus_bindings.setdefault(run_id, [])

        if node_subject not in bindings:
            async def on_node(enveloped: Dict[str, Any]):
                rid = str(enveloped.get("run_id") or run_id)
                node = enveloped.get("node", enveloped)
                extra = {"value": enveloped.get("value"), "best_metric": enveloped.get("best_metric")}
                self.timeline_append_row(rid, node=node, extra=extra)
            await self._evt.add_route(node_subject, on_node)  # type: ignore
            bindings.append(node_subject)

        if report_subject not in bindings:
            async def on_report(enveloped: Dict[str, Any]):
                rid = str(enveloped.get("run_id") or run_id)
                await self.timeline_finalize(rid)
            await self._evt.add_route(report_subject, on_report)  # type: ignore
            bindings.append(report_subject)

    async def timeline_detach_bus(self, run_id: str) -> None:
        for subj in self._bus_bindings.pop(run_id, []):
            await self._evt.remove_route(subj)  # type: ignore

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

        # 🎥 <-- NEW: add one frame per row
        for i in range(matrix.shape[0]):
            row = matrix[i : i + 1, :]
            vpm_out, _ = self._pipeline.run(row, {"enable_gif": False})
            gif.add_frame(
                vpm_out,
                metrics={"step": i, "loss": 1 - row.mean(), "val_loss": row.mean(), "acc": row.std()},
            )

        gif_path = out_path
        gif.save_gif(gif_path, fps=fps)
        _logger.info(f"ZeroModelService: rendered {len(gif.frames)} frames to {gif_path}")

        return {
            "output_path": gif_path,
            "frames": len(gif.frames),
            "shape": list(matrix.shape),
        }
