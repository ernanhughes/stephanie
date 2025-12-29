# stephanie/services/workers/vpm_worker.py

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from stephanie.services.zeromodel_service import ZeroModelService

log = logging.getLogger(__name__)



@dataclass
class VPMWorkerInline:
    """
    Simplified in-process VPM/filmstrip writer (no bus).
    Writes timeline rows via ZeroModelService and saves image artifacts next to the run.

    Conventions:
      - 'append'   : single row of named metrics
      - 'add_channels': multi-row timeseries (namespace-prefixed columns)
      - 'add_vpm'  : save VPM as RGB composite + optional per-channel PNGs
      - 'add_frame': push a single filmstrip frame (later saved by 'save_gif')
      - 'save_gif' : finalize a filmstrip.gif
    """
    zm: Any
    logger: Any = None

    def __post_init__(self):
        self.logger = self.logger or log
        self._frames: Dict[str, List[np.ndarray]] = {}  # run_id -> frames (H,W,3 uint8)

    # -------- timeline rows (numbers) --------
    def append(self, run_id: str, node_id: str, metrics: Dict[str, Any]):
        cols = metrics.get("columns")
        vals = metrics.get("values")

        if (not isinstance(cols, list) or not isinstance(vals, list)) and isinstance(metrics.get("vector"), dict):
            vec = metrics["vector"]
            cols = list(vec.keys())
            vals = [float(vec[k]) for k in cols]

        self.zm.timeline_append_row(run_id, metrics_columns=cols, metrics_values=vals)
        if self.logger:
            self.logger.info(f"[VPMWorkerInline] Row appended for node {node_id} in run {run_id}")

    def add_channels(self, run_id: str, channels: Dict[str, np.ndarray], namespace: str = "vpm"):
        if not channels:
            log.warning(f"[VPMWorkerInline] No channels provided for run {run_id}")
            return
        lengths = [len(arr) for arr in channels.values() if isinstance(arr, np.ndarray)]
        if not lengths:
            log.warning(f"[VPMWorkerInline] No valid numpy arrays in channels for run {run_id}")
            return
        n_timesteps = lengths[0]
        if not all(l == n_timesteps for l in lengths):
            raise ValueError(f"All channels must share length. Got: { {k:len(v) for k,v in channels.items()} }")

        cols = [f"{namespace}.{k}" for k in channels.keys()]
        keys = list(channels.keys())
        for i in range(n_timesteps):
            row_vals = [float(channels[k][i]) for k in keys]
            self.append(run_id=run_id, node_id=f"{namespace}.{i}", metrics={"columns": cols, "values": row_vals})
        if self.logger:
            self.logger.info(f"[VPMWorkerInline] Added {n_timesteps}×{len(channels)} channels under '{namespace}'")

    # -------- image artifacts --------
    def _ensure_dir(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    def add_vpm(self, run_id: str, vpm: np.ndarray, out_dir: Path, name: str, save_channels: bool = False):
        """
        vpm: [C,H,W] (uint8 or float in [0,1])
        Saves: <out_dir>/<name>.png (RGB composite)
               optionally: <out_dir>/<name>_ch{k}.png (per channel)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        X = vpm.astype(np.float32)
        if X.max() <= 1.0: X = X * 255.0
        X = np.clip(X, 0, 255).astype(np.uint8)

        # Compose RGB from first 3 channels (pad if needed)
        C, H, W = X.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(min(3, C)):
            ch = X[i]
            cmin, cmax = ch.min(), ch.max()
            if cmax > cmin:
                ch = ((ch - cmin) * 255.0 / (cmax - cmin)).astype(np.uint8)
            rgb[..., i] = ch

        Image.fromarray(rgb).save(out_dir / f"{name}.png")

        if save_channels:
            for k in range(C):
                Image.fromarray(X[k]).save(out_dir / f"{name}_ch{k}.png")

        if self.logger:
            self.logger.info(f"[VPMWorkerInline] Saved VPM composite '{name}.png' (C={C}) to {out_dir}")

    def add_frame(self, run_id: str, frame_rgb: np.ndarray):
        """frame_rgb: (H,W,3) uint8"""
        self._frames.setdefault(run_id, []).append(frame_rgb)

    def save_gif(self, run_id: str, out_path: Path, fps: int = 1):
        """
        Saves collected frames to an animated GIF.
        Requires imageio; we avoid importing unless needed.
        """
        from imageio.v2 import mimsave
        frames = self._frames.get(run_id, [])
        if not frames:
            log.warning(f"[VPMWorkerInline] No frames to save for run {run_id}")
            return None
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mimsave(out_path, frames, fps=fps, loop=0)
        if self.logger:
            self.logger.info(f"[VPMWorkerInline] Filmstrip saved: {out_path}")
        return str(out_path)

    # -------- finalize / helpers --------
    async def finalize(self, run_id: str, out_path: str):
        res = await self.zm.timeline_finalize(run_id, out_path=out_path)
        if self.logger:
            self.logger.info(f"[VPMWorkerInline] Timeline finalized for run {run_id}")
        return res

class VPMWorker:
    """
    Usage:
        worker = VPMWorker(cfg, memory, container, logger)
        worker.start()                     # from sync init paths
        # or:
        await worker.start_async()         # from async bootstrap
    """
    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        # Subjects (override via cfg.zeromodel.subjects if needed)
        scfg = (self.cfg.get("zeromodel") or {}).get("subjects") or {}
        self.subject_ready = scfg.get("metrics_ready", "arena.metrics.ready")
        self.subject_report = scfg.get("ats_report", "arena.ats.report")

        # ZeroModel service (timeline_open/append don’t need initialize; finalize does)
        self.zm: ZeroModelService = container.get("zeromodel")

        # Track open timelines so we only open once per run
        self._open_runs: set[str] = set()

        # Bus subscription retry logic
        self._subscription_retry_count = 0
        self._max_retries = 5
        self._retry_delay = 1.0  # seconds
        self._health_task = None

    async def start(self) -> None:
        """Start with retry logic for subscriptions"""
        await self._subscribe_with_retry(self.subject_ready, self.handle_metrics_ready)
        await self._subscribe_with_retry(self.subject_report, self.handle_report)

        self.logger.log(
            "VPMWorkerAttached",
            {"subjects": [self.subject_ready, self.subject_report]},
        )
        self._health_task = asyncio.create_task(self._monitor_bus_health())

    async def _subscribe_with_retry(self, subject: str, handler: callable) -> None:
        """Subscribe with exponential backoff retry"""
        try:
            await self.memory.bus.subscribe(subject, handler)
            self._subscription_retry_count = 0
        except Exception as e:
            self._subscription_retry_count += 1
            if self._subscription_retry_count <= self._max_retries:
                delay = self._retry_delay * (2 ** (self._subscription_retry_count - 1))
                log.warning(
                    "Subscription failed for %s (retry %d/%d): %r", subject, self._subscription_retry_count, self._max_retries, e
                )
                await asyncio.sleep(delay)
                await self._subscribe_with_retry(subject, handler)
            else:
                log.error(
                    f"Failed to subscribe to {subject} after {self._max_retries} retries"
                )

    async def handle_metrics_ready(self, payload: Dict[str, Any]) -> None:
        """
        Safely handle metrics.ready messages, even if partial or malformed.
        """
        try:
            run_id = payload.get("run_id")
            node_id = payload.get("node_id")
            if not run_id or not node_id:
                log.warning(f"[VPMWorker] ⚠️ Missing run_id/node_id in payload: {payload}")
                return

            # 🧩 Gracefully handle incomplete payloads
            names = list(payload.get("metrics_columns") or [])
            values = list(payload.get("metrics_values") or [])

            if not names or not values:
                # Try to reconstruct something minimal from results
                scores = payload.get("scores") or {}
                if scores:
                    # Flatten fallback: each scorer.aggregate as metric
                    for scorer, result in scores.items():
                        if "aggregate" in result:
                            names.append(f"{scorer}.aggregate")
                            values.append(result["aggregate"])
                    log.info(
                        f"[VPMWorker] 🧩 Fallback metrics built for node {node_id} ({len(names)} metrics)"
                    )
                else:
                    log.warning(
                        f"[VPMWorker] ⚠️ No metrics found for node {node_id}, skipping."
                    )
                    return  # nothing to append

            # ✅ Append to ZeroModel timeline
            self.zm.timeline_append_row(
                run_id=run_id,
                metrics_columns=names,
                metrics_values=values,
            )
            log.info(f"[VPMWorker] ✅ Row appended for node {node_id}")

        except Exception as e:
            log.error(
                f"[VPMWorker] ❌ handle_metrics_ready failed: {e} | payload={payload}"
            )

    async def handle_report(self, payload: Any) -> None:
        """Handle payload as either dict or bytes"""
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                log.error("Invalid JSON payload")
                return

        if not isinstance(payload, dict):
            log.error("Payload must be dict or JSON string")
            return

        run_id = str(payload.get("run_id") or "")
        if not run_id:
            return

        try:
            # Initialize ZeroModel pipeline lazily for finalize
            if not getattr(self.zm, "_initialized", False):
                await self.zm.initialize()

            res = await self.zm.timeline_finalize(run_id, out_path=payload.get("out_path"))
            log.info("VPMFinalized run_id %s %s", run_id, str(res))
            self._open_runs.discard(run_id)

        except Exception as e:
            log.error(
                "VPMFinalizeError error %s | trace:%s | run_id: %s", str(e),  traceback.format_exc(), run_id
            )

    async def _monitor_bus_health(self):
        """Continuously monitor bus health with detailed logging"""
        while True:
            try:
                health = self.memory.bus.health_check()
                bus_type = health["bus_type"]
                status = health["status"]
                details = health["details"]
                
                # Format detailed health info
                if bus_type == "nats":
                    details_str = (
                        f"connected={details.get('connected', '?')}, "
                        f"closed={details.get('closed', '?')}, "
                        f"uptime={details.get('connection_uptime', 0):.1f}s, "
                        f"reconnects={details.get('reconnect_attempts', 0)}"
                    )
                elif bus_type == "inproc":
                    details_str = (
                        f"subscriptions={details.get('subscriptions', '?')}, "
                        f"mem={details.get('memory_usage', '?')}"
                    )
                else:
                    details_str = str(details)
                
                # Log with color-coded status
                if status :
                    log.debug("BUS HEALTH: 🟢 %s - %s | %s", bus_type, status, details_str)
                elif status == "disconnected":
                    log.warning("BUS HEALTH: 🔴 %s - %s | %s", bus_type, status, details_str)
                await asyncio.sleep(30)
            except Exception as e:
                log.error("BUS HEALTH CHECK FAILED: %s", str(e))
                await asyncio.sleep(10)