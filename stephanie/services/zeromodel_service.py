# stephanie/services/zero_model_service.py
from __future__ import annotations

import os, json, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from stephanie.services.service_protocol import Service
from stephanie.services.event_service import EventService
from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.tools.gif_logger import GifLogger  # ZeroModel owns rendering

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

    def timeline_append_row(self, run_id: str, *, node: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
        """Append one row derived from a node event (fast, in-process)."""
        sess = self._sessions.get(run_id)
        if not sess:
            return
        extra = extra or {}
        act = node.get("type","draft")
        row = [
            float(node.get("metric") or 0.0),          # metric (primary)
            float(extra.get("value") or 0.0),          # value/ucb mean
            float(node.get("visits") or 1.0),          # visits
            1.0 if node.get("bug") else 0.0,           # bug flag
            1.0 if act == "draft"   else 0.0,
            1.0 if act == "improve" else 0.0,
            1.0 if act == "debug"   else 0.0,
        ]
        # shape safety
        if len(row) < len(sess.metrics_order):
            row += [0.0]*(len(sess.metrics_order)-len(row))
        elif len(row) > len(sess.metrics_order):
            row = row[:len(sess.metrics_order)]
        sess.rows.append(row)
        sess.meta.append({
            "node_id": node.get("id"),
            "parent_id": node.get("parent_id"),
            "type": act,
            "ts": time.time(),
        })

    async def timeline_finalize(self, run_id: str, *, fps: Optional[int] = None, datestamp: bool = True) -> Dict[str, Any]:
        """Close session, render GIF, emit bus event, and persist companion JSON."""
        sess = self._sessions.pop(run_id, None)
        if not sess:
            return {"status": "noop", "reason": "no_session"}

        mat = sess.as_matrix()  # (steps, metrics)
        base = os.path.join(sess.out_dir, f"vpm_timeline_{run_id}")
        # guarantee .gif at the call site
        gif_path = base if base.lower().endswith(".gif") else base + ".gif"

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
        metrics: Optional[List[str]] = None,   # kept for signature compat; not used by your executor
        options: Optional[Dict[str, Any]] = None,
        datestamp: bool = False,
    ) -> Dict[str, Any]:
        """
        Render a timeline GIF by delegating to ZeroModel's PipelineExecutor.run(vpm, context).
        We give it the matrix as the VPM and a prepared context that includes a GifLogger,
        target gif path, and fps. The executor will add frames and save to gif_path.
        """
        assert self._pipeline is not None, "ZeroModelService not initialized"

        # --- ensure .gif extension ---
        root, ext = os.path.splitext(out_path)
        if not ext:
            out_path = root + ".gif"
            root, ext = os.path.splitext(out_path)

        # --- datestamp prior to saving (keep .gif extension) ---
        if datestamp:
            out_path = f"{root}_{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}{ext}"

        # --- empty-matrix guard (executor expects a VPM-like ndarray) ---
        if not isinstance(matrix, np.ndarray) or matrix.size == 0:
            # 1 x max(1,N) zero VPM to produce a valid (minimal) GIF
            ncols = 1 if metrics is None else max(1, len(metrics))
            matrix = np.zeros((1, ncols), dtype=np.float32)

        # Prepare context for the executor. It will:
        #   - create its own GifLogger if none present,
        #   - but we provide one so we keep control over max_frames, etc.
        gif = GifLogger(max_frames=self._max_frames)
        ctx: Dict[str, Any] = {
            **(options or {}),
            "gif_logger": gif,
            "gif_path": out_path,           # <-- REQUIRED so executor saves it
            "gif_fps": (fps or self._gif_fps),
            "enable_gif": True,
            # Optional debug stripe keys supported by your executor helpers
            "gif_debug_stripe": True,
            "gif_debug_stripe_label": "ATS",
        }

        # Run the pipeline. It will add frames and save to gif_path when finishing.
        vpm_out, ctx_out = self._pipeline.run(matrix, ctx)

        # If for any reason the executor didn’t save, save here as a fallback.
        saved_path = ctx_out.get("gif_saved")
        if not saved_path:
            try:
                gif.save_gif(out_path, fps=(fps or self._gif_fps))
                saved_path = out_path
            except Exception as _:
                # leave saved_path as None
                pass

        return {
            "output_path": saved_path or out_path,
            "frames": len(gif.frames),
            "shape": list(vpm_out.shape),
        }
