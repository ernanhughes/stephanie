# stephanie/tools/metrics_enqueue_tool.py
from __future__ import annotations

import time
from typing import Any, Dict

from stephanie.constants_metrics import METRICS_REQ
from stephanie.services.service_protocol import Service


class MetricsEnqueueTool(Service):
    """
    Tiny 'tool' (not an agent). Publishes a metrics job and returns.
    """

    def __init__(self, cfg, memory, logger, container=None):
        super().__init__(cfg, memory, logger)
        self.container = container
        self._evt = None

    async def initialize(self, **kwargs):
        self._evt = self.container.get("event")
        assert self._evt, "EventService required"

    async def health_check(self):
        return {"status":"ok"}

    async def shutdown(self): 
        pass

    def enqueue(self, *, run_id: str, node: Dict[str, Any], goal_text: str, prompt_text: str, extras: Dict[str, Any] | None = None) -> None:
        """
        Fire-and-forget. No await, no compute.
        """
        payload = {
            "run_id": run_id,
            "node_id": str(node.get("id")),
            "parent_id": str(node.get("parent_id", "")),
            "stage_name": node.get("stage", ""),
            "agent_name": node.get("agent", ""),
            "prompt_text": (prompt_text or "").strip(),
            "goal_text": (goal_text or "").strip(),
            "best_metric": node.get("metric"),
            "ts_enqueued": time.time(),
            "extras": extras or {},
        }
        # publish synchronously; EventService should queue/dispatch
        self._evt.publish_sync(METRICS_REQ, payload)  # add publish_sync to EventService if not present
