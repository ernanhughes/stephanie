from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Any, Dict, Optional
import numpy as np
from typing import Dict, Optional

from stephanie.services.zeromodel_service import ZeroModelService

_logger = logging.getLogger(__name__)


class VPMWorkerInline:
    """Simplified in-process timeline writer (no bus)."""
    def __init__(self, zm: ZeroModelService, logger=None):
        self.zm = zm
        self.logger = logger or _logger

    def append(self, run_id: str, node_id: str, metrics: Dict[str, Any]):
        cols = metrics.get("columns")
        vals = metrics.get("values")

        if (not isinstance(cols, list) or not isinstance(vals, list)) and isinstance(metrics.get("vector"), dict):
            vec = metrics["vector"]
            cols = list(vec.keys())
            vals = [float(vec[k]) for k in cols]

        self.zm.timeline_append_row(
            run_id,
            metrics_columns=cols,
            metrics_values=vals,
        )
        self.logger.log(f"[VPMWorkerInline] Row appended for node {node_id} in run {run_id}")

    async def finalize(self, run_id: str, out_path: str):
        res = await self.zm.timeline_finalize(
            run_id,
            out_path=out_path,
        )
        self.logger.info(f"[VPMWorkerInline] Timeline finalized for run {run_id}")
        return res

    def _log_progress(self, stage, i, total):
        try:
            pct = int(100 * i / max(total,1))
            self.logger.log("TopologyProgress", {"stage": stage, "i": i, "total": total, "pct": pct})
        except Exception:
            pass

    def add_channels(self, run_id: str, channels: Dict[str, np.ndarray], namespace: str = "hall"):
        """
        Append a batch of VPM channels as timeline columns.

        Args:
            run_id (str): The run identifier.
            channels (Dict[str, np.ndarray]): Dictionary of named 1D arrays (e.g., {"R": [...], "G": [...]})
                Each array must be 1D and of the same length (timesteps).
            namespace (str): Prefix for column names (e.g., "hall" → "hall.R", "hall.G").

        Example:
            add_channels("run-123", {"R": [0.1, 0.8], "G": [0.3, 0.2]}, "hall")
            → Appends two rows:
                Row 0: {"hall.R": 0.1, "hall.G": 0.3}
                Row 1: {"hall.R": 0.8, "hall.G": 0.2}
        """
        if not channels:
            self.logger.warning(f"[VPMWorkerInline] No channels provided for run {run_id}")
            return

        # Validate: all channels must be 1D numpy arrays of same length
        lengths = [len(arr) for arr in channels.values() if isinstance(arr, np.ndarray)]
        if not lengths:
            self.logger.warning(f"[VPMWorkerInline] No valid numpy arrays in channels for run {run_id}")
            return

        n_timesteps = lengths[0]
        if not all(l == n_timesteps for l in lengths):
            raise ValueError(
                f"All channels must have the same length. Got lengths: {dict(zip(channels.keys(), lengths))}"
            )

        # Build column names with namespace prefix
        cols = [f"{namespace}.{k}" for k in channels.keys()]

        # Convert arrays to list of floats
        vals_list = []
        for i in range(n_timesteps):
            row_vals = [float(channels[k][i]) for k in channels.keys()]
            vals_list.append(row_vals)

        # Append each timestep as a row
        for i, vals in enumerate(vals_list):
            self.append(
                run_id=run_id,
                node_id=f"vpm.{namespace}.{i}",  # Optional: node_id to trace position
                metrics={"columns": cols, "values": vals},
            )

        self.logger.info(f"[VPMWorkerInline] Added {n_timesteps} timesteps of {len(channels)} channels under namespace '{namespace}' for run {run_id}")

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
                _logger.warning(
                    "Subscription failed for %s (retry %d/%d): %r", subject, self._subscription_retry_count, self._max_retries, e
                )
                await asyncio.sleep(delay)
                await self._subscribe_with_retry(subject, handler)
            else:
                _logger.error(
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
                self.logger.warning(f"[VPMWorker] ⚠️ Missing run_id/node_id in payload: {payload}")
                return

            # 🧩 Gracefully handle incomplete payloads
            names = list(payload.get("metrics_columns") or [])
            values = list(payload.get("metrics_values") or [])
            vector = payload.get("metrics_vector", {})

            if not names or not values:
                # Try to reconstruct something minimal from results
                scores = payload.get("scores") or {}
                if scores:
                    # Flatten fallback: each scorer.aggregate as metric
                    for scorer, result in scores.items():
                        if "aggregate" in result:
                            names.append(f"{scorer}.aggregate")
                            values.append(result["aggregate"])
                    _logger.info(
                        f"[VPMWorker] 🧩 Fallback metrics built for node {node_id} ({len(names)} metrics)"
                    )
                else:
                    _logger.warning(
                        f"[VPMWorker] ⚠️ No metrics found for node {node_id}, skipping."
                    )
                    return  # nothing to append

            # ✅ Append to ZeroModel timeline
            self.zm.timeline_append_row(
                run_id=run_id,
                metrics_columns=names,
                metrics_values=values,
            )
            _logger.info(f"[VPMWorker] ✅ Row appended for node {node_id}")

        except Exception as e:
            _logger.error(
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
                _logger.error("Invalid JSON payload")
                return

        if not isinstance(payload, dict):
            _logger.error("Payload must be dict or JSON string")
            return

        run_id = str(payload.get("run_id") or "")
        if not run_id:
            return

        try:
            # Initialize ZeroModel pipeline lazily for finalize
            if not getattr(self.zm, "_initialized", False):
                await self.zm.initialize()

            res = await self.zm.timeline_finalize(run_id, out_path=payload.get("out_path"))
            _logger.info("VPMFinalized run_id %s %s", run_id, str(res))
            self._open_runs.discard(run_id)

        except Exception as e:
            _logger.error(
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
                    _logger.debug("BUS HEALTH: 🟢 %s - %s | %s", bus_type, status, details_str)
                elif status == "disconnected":
                    _logger.warning("BUS HEALTH: 🔴 %s - %s | %s", bus_type, status, details_str)
                await asyncio.sleep(30)
            except Exception as e:
                _logger.error("BUS HEALTH CHECK FAILED: %s", str(e))
                await asyncio.sleep(10)