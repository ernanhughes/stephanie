# stephanie/services/reporting_service.py
from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from stephanie.services.service_protocol import Service


# --- Utility Functions ---
def _truncate(v, max_len=400):
    if isinstance(v, str):
        return (v[:max_len] + "…") if len(v) > max_len else v
    return v

def _safe(obj: Any, max_str=400) -> Any:
    if obj is None or isinstance(obj, (int, float, bool)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, str):
        return _truncate(obj, max_str)
    if isinstance(obj, (list, tuple)):
        return [_safe(x, max_str) for x in obj[:50]]
    if isinstance(obj, dict):
        return {str(k): _safe(v, max_str) for k, v in list(obj.items())[:100]}
    try:
        return _truncate(str(obj), max_str)
    except Exception:
        return None


# --- Sink Interfaces ---
class BaseSink:
    async def emit(self, event: Dict[str, Any]):  # override
        pass
    async def flush(self):
        pass

class JsonlSink(BaseSink):
    def __init__(self, path: str):
        self.path = path
        self._lock = asyncio.Lock()
    async def emit(self, event: Dict[str, Any]):
        line = json.dumps(event, ensure_ascii=False)
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

class LoggerSink(BaseSink):
    def __init__(self, logger):
        self.logger = logger
    async def emit(self, event: Dict[str, Any]):
        msg = {k: event[k] for k in ("ts","run_id","agent","stage","note") if k in event}
        try:
            agent = msg.get("agent") 
            if agent == "UnknownAgent":
                self.logger.info("LogReport", msg)
            self.logger.log("CBRReport", msg)
        except Exception:
            pass


# --- Reporting Service ---
class ReportingService(Service):
    """
    Service that emits structured events to sinks (JSONL, logger, etc.)
    and appends compact entries into context["REPORTS"] for ReportFormatter.
    """

    def __init__(self, sinks: List[BaseSink], enabled: bool = True, sample_rate: float = 1.0):
        self.enabled = enabled
        self.sinks = sinks
        self.sample_rate = float(sample_rate)

        self._q: asyncio.Queue = asyncio.Queue(maxsize=2048)
        self._consumer: Optional[asyncio.Task] = None
        self._initialized = False

    # === Service Protocol ===
    def initialize(self, **kwargs) -> None:
        """Start background consumer loop for reporting."""
        if not self._initialized:
            loop = asyncio.get_event_loop()
            self._consumer = loop.create_task(self._run())
            self._initialized = True

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "queue_size": self._q.qsize(),
                "queue_capacity": self._q.maxsize,
                "sinks": len(self.sinks),
                "consumer_running": bool(self._consumer and not self._consumer.done()),
            },
            "dependencies": {},
        }

    def shutdown(self) -> None:
        """Stop consumer and clear queue."""
        if self._consumer:
            self._consumer.cancel()
            self._consumer = None
        self._initialized = False
        self._q = asyncio.Queue(maxsize=2048)

    @property
    def name(self) -> str:
        return "reporting-service-v1"

    # === Internal Worker ===
    async def _run(self):
        try:
            while True:
                ev = await self._q.get()
                for s in self.sinks:
                    try:
                        await s.emit(ev)
                    except Exception:
                        pass
        except asyncio.CancelledError:
            pass

    # === Public API ===
    async def emit(
        self,
        *,
        context: Dict[str,Any],
        stage: str,
        status: Optional[str] = None,
        summary: Optional[str] = None,
        finalize: bool = False,
        **payload
    ):
        """
        stage:     logical stage name
        status:    'running' | 'done' | 'error'
        summary:   short line for the report
        finalize:  mark the stage as finished
        payload:   free-form fields (kept small)
        """
        if not self.enabled:
            return

        g = context.get("goal") or {}
        event = {
            "ts": time.time(),
            "run_id": context.get("run_id") or g.get("run_id") or context.get("pipeline_run_id") or str(uuid.uuid4()),
            "plan_trace_id": context.get("plan_trace_id"),
            "goal_id": g.get("id"),
            "agent": context.get("agent_name") or context.get("AGENT_NAME") or "UnknownAgent",
            "stage": stage,
            **_safe(payload),
        }

        # Append to context report
        try:
            self._append_ctx_report(context, event, status=status, summary=summary, finalize=finalize)
        except Exception:
            pass

        # Enqueue for sinks
        try:
            self._q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # drop silently

    # === Context Reporting ===
    def _append_ctx_report(
        self,
        ctx: Dict[str,Any],
        event: Dict[str,Any],
        status: Optional[str],
        summary: Optional[str],
        finalize: bool
    ):
        reports = ctx.setdefault("REPORTS", [])
        stage_name = event.get("stage") or "stage"
        agent = event.get("agent") or "UnknownAgent"
        now_iso = datetime.now(timezone.utc).isoformat()

        row = None
        for r in reports:
            if r.get("stage") == stage_name and r.get("agent") == agent:
                row = r
                break

        if row is None:
            row = {
                "stage": stage_name,
                "agent": agent,
                "status": status or "running",
                "summary": summary or "",
                "start_time": now_iso,
                "end_time": None,
                "entries": [],
            }
            reports.append(row)
        else:
            if status:
                row["status"] = status
            if summary:
                row["summary"] = summary

        entry_event = event.get("event") or event.get("note") or "event"
        entry_payload = {k: v for k, v in event.items()
                         if k not in {"ts","run_id","plan_trace_id","goal_id","agent","stage"}}

        row["entries"].append({
            "event": entry_event,
            **entry_payload
        })

        row["end_time"] = now_iso
        if finalize:
            row["status"] = status or row.get("status") or "done"
