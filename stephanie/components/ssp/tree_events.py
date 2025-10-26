from __future__ import annotations
from typing import Dict, Any
from stephanie.components.ssp.util import PlanTrace_safe
from stephanie.utils.trace_logger import trace_logger
from stephanie.utils.json_sanitize import sanitize

class TreeEventEmitter:
    async def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        meta = sanitize({"event": event, **(payload or {})})
        trace_logger.log(PlanTrace_safe(
            trace_id=f"tree-{event}",
            role="solver",
            goal="tree-event",
            status="progress" if event in {"start", "progress"} else "completed",
            metadata=meta,
            input="",
            output=f"[tree] {event}",
            artifacts={}
        ))
