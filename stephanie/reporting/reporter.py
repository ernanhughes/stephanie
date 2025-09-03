# stephanie/reporting/reporter.py
from __future__ import annotations
import asyncio, json, time, math, uuid
from typing import Any, Dict, Optional
from datetime import datetime, timezone

def _truncate(v, max_len=400):
    if isinstance(v, str):
        return (v[:max_len] + "…") if len(v) > max_len else v
    return v

def _safe(obj: Any, max_str=400) -> Any:
    # sanitize for JSON + avoid huge payloads
    if obj is None or isinstance(obj, (int, float, bool)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, str):
        return _truncate(obj, max_str)
    if isinstance(obj, (list, tuple)):
        return [_safe(x, max_str) for x in obj[:50]]  # cap list length
    if isinstance(obj, dict):
        return {str(k): _safe(v, max_str) for k, v in list(obj.items())[:100]}
    try:
        return _truncate(str(obj), max_str)
    except Exception:
        return None

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
        # keep logs tiny
        msg = {k: event[k] for k in ("ts","run_id","agent","stage","note") if k in event}
        try:
            # assuming your logger has .log(event_type, payload)
            self.logger.log("CBRReport", msg)
        except Exception:
            pass

class Reporter:
    """
    Efficient Reporter with two responsibilities:
      1) Emit structured events to sinks (JSONL, logger, etc.)
      2) Append compact entries into context["REPORTS"] for ReportFormatter
    """
    def __init__(self, sinks: list[BaseSink], enabled: bool = True, sample_rate: float = 1.0):
        self.enabled = enabled
        self.sinks = sinks
        self.sample_rate = float(sample_rate)
        self._q: asyncio.Queue = asyncio.Queue(maxsize=2048)
        self._consumer: Optional[asyncio.Task] = None

    async def start(self):
        if self._consumer is None:
            self._consumer = asyncio.create_task(self._run())

    async def stop(self):
        if self._consumer:
            self._consumer.cancel()
            self._consumer = None

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

    # --------- PUBLIC API ---------
    async def emit(self, *, ctx: Dict[str,Any], stage: str, status: Optional[str] = None,
                   summary: Optional[str] = None, finalize: bool = False, **payload):
        """
        stage:     logical stage name (will be the section header in the report)
        status:    optional 'running'|'done'|'error' (shown in report)
        summary:   optional short line under the stage header
        finalize:  mark the stage as finished (sets end_time & status if provided)
        payload:   free-form small fields (kept tiny & safe)
        """
        if not self.enabled:
            return

        g = ctx.get("goal") or {}
        event = {
            "ts": time.time(),
            "run_id": ctx.get("run_id") or g.get("run_id") or ctx.get("pipeline_run_id") or str(uuid.uuid4()),
            "plan_trace_id": ctx.get("plan_trace_id"),
            "goal_id": g.get("id"),
            "agent": ctx.get("agent_name") or ctx.get("AGENT_NAME") or "UnknownAgent",
            "stage": stage,
            **_safe(payload),
        }

        # (A) append to in-memory report on the context (for ReportFormatter)
        try:
            self._append_ctx_report(ctx, event, status=status, summary=summary, finalize=finalize)
        except Exception:
            pass

        # (B) queue for sinks
        try:
            self._q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # drop silently to keep fast path fast

    # --------- INTERNAL: context["REPORTS"] appender ---------
    def _append_ctx_report(self, ctx: Dict[str,Any], event: Dict[str,Any],
                           status: Optional[str], summary: Optional[str], finalize: bool):
        """
        Build/append a stage block in ctx["REPORTS"] matching ReportFormatter expectations.
        """
        reports = ctx.setdefault("REPORTS", [])
        stage_name = event.get("stage") or "stage"
        agent = event.get("agent") or "UnknownAgent"
        now_iso = datetime.now(timezone.utc).isoformat()

        # find existing stage entry for (stage, agent)
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

        # Create a compact entry; prefer 'event' or 'note' fields if provided
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
