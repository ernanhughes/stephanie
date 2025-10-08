from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Any, Dict, Optional

from stephanie.services.zeromodel_service import ZeroModelService

_logger = logging.getLogger(__name__)

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
                    f"Subscription failed for {subject} (retry {self._subscription_retry_count}/{self._max_retries}): {e}"
                )
                await asyncio.sleep(delay)
                await self._subscribe_with_retry(subject, handler)
            else:
                _logger.error(
                    f"Failed to subscribe to {subject} after {self._max_retries} retries"
                )

    async def handle_metrics_ready(self, payload: Dict[str, Any]) -> None:
        """Handle completed metrics payloads and record them in the ZeroModel timeline."""
        run_id = str(payload.get("run_id") or "")
        node_id = str(payload.get("node_id") or "")
        if not run_id:
            return

        try:
            # Ensure the timeline session is open once per run
            if run_id not in self._open_runs:
                self.zm.timeline_open(run_id)
                self._open_runs.add(run_id)

            # --- 1️⃣ Extract metrics consistently ---

            names = list(payload["metrics_columns"])
            values = [float(v) for v in payload.get("metrics_values", [])]
            names = ["node_id"] + names
            values = [node_id] + values


            # --- 2️⃣ Send full set to ZeroModel ---
            self.zm.timeline_append_row(
                run_id,
                metrics_columns=names,
                metrics_values=values, 
            )

            self.logger.log(
                "VPMRowAppended",
                {"run_id": run_id, "node_id": node_id, "metrics_count": len(values)},
            )

        except Exception as e:
            self.logger.log(
                "VPMMetricsReadyError",
                {"error": str(e), "trace": traceback.format_exc(), "run_id": run_id, "node_id": node_id},
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

            res = await self.zm.timeline_finalize(run_id)
            self.logger.log("VPMFinalized", {"run_id": run_id, **res})
            self._open_runs.discard(run_id)

        except Exception as e:
            self.logger.log(
                "VPMFinalizeError",
                {"error": str(e), "trace": traceback.format_exc(), "run_id": run_id},
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
                    _logger.debug(f"BUS HEALTH: 🟢 {bus_type} - {status} | {details_str}")
                elif status == "disconnected":
                    _logger.warning(f"BUS HEALTH: 🔴 {bus_type} - {status} | {details_str}")
                await asyncio.sleep(30)
            except Exception as e:
                _logger.error(f"BUS HEALTH CHECK FAILED: {str(e)}")
                await asyncio.sleep(10)