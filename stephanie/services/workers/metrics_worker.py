# stephanie/workers/metrics_worker.py
"""
MetricsWorker
-------------
Consumes 'metrics.request' jobs, computes metrics with MetricsService,
and publishes 'metrics.ready' with the vector + parts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from typing import Any, Dict, Optional

class MetricsWorker:
    """
    Usage:
        worker = MetricsWorker(cfg, memory, container, logger)
        worker.start()
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger or logging.getLogger(__name__)

        # subjects (override in cfg if you like)
        mcfg = (cfg.get("metrics") or {}).get("subjects") or {}
        self.subject_req = mcfg.get("request", "arena.metrics.request")
        self.subject_ready = mcfg.get("ready", "arena.metrics.ready")

        self.subject = "arena.metrics.request"
        self.queue_group = "metrics"

        # service lives inside worker (sync init; service does async inside its methods)
        self.metrics_service = container.get("metrics")

    async def handle_job(self, envelope: Dict[str, Any]):
        """
        Expected payload:
          {
            "run_id": "...",
            "node_id": "...",
            "goal_text": "...",
            "prompt_text": "...",
            ... (free-form extras ok)
          }
        """
        envelope = None
        try:
            run_id = str(envelope.get("run_id", "") if isinstance(envelope, dict) else "")
            node_id = str(envelope.get("node_id", ""))
            goal_text = envelope.get("goal_text", "") or ""
            prompt_text = envelope.get("prompt_text", "") or ""

            if not prompt_text:
                self.logger.log("MetricsJobSkipped", {"reason": "no_prompt_text", "run_id": run_id, "node_id": node_id})
                return

            self.logger.log("MetricsJobReceived", {"run_id": run_id, "node_id": node_id})

            t0 = time.time()
            res = await self.metrics_service.build_prompt_metrics(
                goal_text=goal_text,
                prompt_text=prompt_text,
                context={"run_id": run_id, "node_id": node_id},
            )
            elapsed = time.time() - t0

            out = {
                **envelope,
                "metrics_vector": res.get("vector"),
                "metrics_parts": res.get("parts"),
                "duration_sec": elapsed,
                "ts_completed": time.time(),
            }

            await self.memory.bus.publish(
                subject=self.subject_ready,
                payload=json.dumps(out).encode("utf-8"),
            )

            self.logger.log("MetricsJobDone", {
                "run_id": run_id,
                "node_id": node_id,
                "dim": len((res.get("vector") or {}).get("names") or []),
                "duration_sec": elapsed,
            })

        except Exception as e:
            self.logger.log("MetricsJobError", {
                "error": str(e),
                "trace": traceback.format_exc(),
                "run_id": envelope.get("run_id", "unknown") if isinstance(envelope, dict) else "unknown",
                "node_id": envelope.get("node_id", "unknown") if isinstance(envelope, dict) else "unknown",
            })

    async def start(self):
        # proper await on async subscribe
        await self.memory.bus.subscribe(
            self.subject,
            self.handle_job,          # your handler (async def ...)
        )
        self.logger.log("MetricsWorkerAttached", {
            "subject": self.subject,
        })

