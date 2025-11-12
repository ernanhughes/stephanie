# stephanie/services/reporting_service.py
from __future__ import annotations

import asyncio
import json
import math
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from stephanie.services.service_protocol import Service
from stephanie.utils.time_utils import now_iso


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
    async def flush(self):
        # File writes are sync/atomic per emit; nothing to flush.
        return

class LoggerSink(BaseSink):
    def __init__(self, logger, level: str = "INFO"):
        self.logger = logger
        self.level = level.upper()
        self._level_map = {
            "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50
        }
    async def emit(self, event: Dict[str, Any]):
        # Keep log concise but useful; include core keys and summary/note if present.
        base = {k: event.get(k) for k in ("ts","run_id","agent","stage","status","summary")}
        note = event.get("note")
        if note: base["note"] = note
        try:
            lvl = self._level_map.get(event.get("level","").upper(), self._level_map.get(self.level, 20))
            self.logger.log(lvl, "ReportEvent", extra={"event": base})
        except Exception:
            # best-effort fallback
            try:
                self.logger.info("ReportEvent %s", base)
            except Exception:
                pass
    async def flush(self):
        return


# --- Reporting Service ---
class ReportingService(Service):
    """
    Service that emits structured events to sinks (JSONL, logger, etc.)
    and appends compact entries into context["REPORTS"] for ReportFormatter.
    """

    def __init__(self, sinks: List[BaseSink], enabled: bool = True, sample_rate: float = 1.0):
        self.enabled = enabled
        self.sinks = sinks
        self.sample_rate = float(sample_rate or 1.0)

        self._q: asyncio.Queue = asyncio.Queue(maxsize=2048)
        self._consumer: Optional[asyncio.Task] = None
        self._initialized = False

    # === Service Protocol ===
    def initialize(self, **kwargs) -> None:
        """Start background consumer loop for reporting."""
        if self._initialized or not self.enabled:
            self._initialized = True
            return
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            self._consumer = loop.create_task(self._run())
            self._initialized = True
        except Exception:
            # If we can't start consumer, keep service disabled gracefully
            self.enabled = False
            self._initialized = False

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if (self._initialized and self.enabled) else "unhealthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "queue_size": self._q.qsize(),
                "queue_capacity": self._q.maxsize,
                "sinks": len(self.sinks),
                "consumer_running": bool(self._consumer and not self._consumer.done()),
                "sample_rate": self.sample_rate,
            },
            "dependencies": {},
        }

    def shutdown(self) -> None:
        """Drain queue, stop consumer, and flush sinks."""
        async def _drain_and_stop():
            # Drain queue
            try:
                while not self._q.empty():
                    ev = self._q.get_nowait()
                    for s in self.sinks:
                        try:
                            await s.emit(ev)
                        except Exception:
                            pass
            except Exception:
                pass
            # Flush sinks
            for s in self.sinks:
                try:
                    await s.flush()
                except Exception:
                    pass
            # Cancel consumer
            if self._consumer:
                self._consumer.cancel()
                try:
                    await self._consumer
                except asyncio.CancelledError:
                    pass

        if self._consumer:
            try:
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(_drain_and_stop())
                else:
                    loop.run_until_complete(_drain_and_stop())
            except Exception:
                pass

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
    def emit_sync(
        self,
        *,
        context: Optional[Dict[str,Any]] = None,
        stage: str = "stage",
        status: Optional[str] = None,
        summary: Optional[str] = None,
        finalize: bool = False,
        **payload
    ):
        """Convenience wrapper for sync callers; schedules async emit."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(context=context or {}, stage=stage, status=status,
                                       summary=summary, finalize=finalize, **payload))
        except RuntimeError:
            # No running loop; start a temp loop to execute
            asyncio.run(self.emit(context=context or {}, stage=stage, status=status,
                                  summary=summary, finalize=finalize, **payload))

    async def emit(
        self,
        *,
        context: Optional[Dict[str,Any]],
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

        # sampling
        if self.sample_rate < 1.0 and random.random() > self.sample_rate:
            return

        ctx = context or {}
        g = (ctx.get("goal") or {}) if isinstance(ctx, dict) else {}
        now = time.time()
        event = {
            "ts": now,
            "ts_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "run_id": ctx.get("run_id") or g.get("run_id") or ctx.get("pipeline_run_id") or str(uuid.uuid4()),
            "plan_trace_id": ctx.get("plan_trace_id"),
            "goal_id": g.get("id"),
            "agent": ctx.get("agent_name") or ctx.get("AGENT_NAME") or "UnknownAgent",
            "stage": stage,
            "status": status or "running",
            "summary": summary or "",
            **_safe(payload),
        }

        # Append to context report
        try:
            if isinstance(ctx, dict):
                self._append_ctx_report(ctx, event, status=status, summary=summary, finalize=finalize)
        except Exception:
            pass

        # Enqueue for sinks (drop-oldest-on-full)
        try:
            self._q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                _ = self._q.get_nowait()  # drop oldest
            except Exception:
                pass
            try:
                self._q.put_nowait(event)
            except Exception:
                pass

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
                "start_time": now_iso(),
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
                         if k not in {"ts","ts_iso","run_id","plan_trace_id","goal_id","agent","stage","status","summary"}}

        row["entries"].append({
            "event": entry_event,
            **entry_payload
        })

        row["end_time"] = now_iso()
        if finalize:
            row["status"] = status or row.get("status") or "done"
