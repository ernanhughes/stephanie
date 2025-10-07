# stephanie/workers/vpm_worker.py
"""
VPMWorker
---------
Consumes 'arena.metrics.ready' and appends rows to ZeroModel timelines.
Consumes 'arena.ats.report' and finalizes the GIF + companion JSON.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Dict, Optional, Set

from stephanie.services.zeromodel_service import ZeroModelService


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
        self.logger = logger or logging.getLogger(__name__)

        # Subjects (override via cfg.zeromodel.subjects if needed)
        scfg = (self.cfg.get("zeromodel") or {}).get("subjects") or {}
        self.subject_ready = scfg.get("metrics_ready", "arena.metrics.ready")
        self.subject_report = scfg.get("ats_report", "arena.ats.report")

        # ZeroModel service (timeline_open/append don’t need initialize; finalize does)
        self.zm = ZeroModelService(cfg=self.cfg, memory=self.memory, logger=self.logger)

        # Track open timelines so we only open once per run
        self._open_runs: Set[str] = set()

    # ---------- lifecycle ----------

    async def start(self) -> None:
        # bus.subscribe(subject, handler) is async; no queue_group support in your bus protocol
        await self.memory.bus.subscribe(self.subject_ready, self.handle_metrics_ready)
        await self.memory.bus.subscribe(self.subject_report, self.handle_report)

        self.logger.log(
            "VPMWorkerAttached",
            {"subjects": [self.subject_ready, self.subject_report]},
        )

    # ---------- handlers ----------

    async def handle_metrics_ready(self, payload: Dict[str, Any]) -> None:
        """
        Payload shape (dict):
          {
            "run_id": "uuid",
            "node_id": "nX",
            "parent_id": "nY" | null,
            "action_type": "draft" | "improve" | "debug",
            "best_metric": 0.73,                 # optional, scalar
            "bug": false,                        # optional
            "metrics_vector": {"names":[...], "values":[...]}  # optional
          }
        """
        run_id = str(payload.get("run_id") or "")
        node_id = str(payload.get("node_id") or "")

        try:
            if not run_id:
                return

            # Open timeline once per run
            if run_id not in self._open_runs:
                # If you want a custom metric order, pass metrics=[...] here
                self.zm.timeline_open(run_id)
                self._open_runs.add(run_id)

            # Build node fields ZeroModel expects
            node = {
                "id": node_id,
                "parent_id": payload.get("parent_id"),
                "type": payload.get("action_type", "draft"),
                "metric": payload.get("best_metric"),
                "visits": 1,
                "bug": bool(payload.get("bug", False)),
            }

            # Choose a scalar value lane (e.g., mean of provided vector)
            vec = payload.get("metrics_vector") or {}
            vals = vec.get("values") or []
            value = float(sum(vals) / len(vals)) if vals else 0.0

            self.zm.timeline_append_row(
                run_id,
                node=node,
                extra={"value": value, "best_metric": node["metric"]},
            )

            self.logger.log(
                "VPMRowAppended",
                {"run_id": run_id, "node_id": node_id, "value": value, "metric": node["metric"]},
            )

        except Exception as e:
            self.logger.log(
                "VPMMetricsReadyError",
                {
                    "error": str(e),
                    "trace": traceback.format_exc(),
                    "run_id": run_id or "unknown",
                    "node_id": node_id or "unknown",
                },
            )

    async def handle_report(self, payload: Dict[str, Any]) -> None:
        """
        Payload shape:
          {"run_id": "uuid"}
        """
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
