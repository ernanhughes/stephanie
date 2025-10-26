# stephanie/utils/trace_logger.py

from __future__ import annotations

import dataclasses
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

from stephanie.utils.json_sanitize import sanitize, dumps_safe

# Optional: import your PlanTrace if present
try:
    from stephanie.traces.plan_trace import PlanTrace  # type: ignore
except Exception:
    PlanTrace = None  # type: ignore


class _PlanTraceLike(Protocol):
    """Structural typing for PlanTrace-like objects."""
    # common fields used across your codebase
    trace_id: str
    role: str
    goal: str
    status: str
    metadata: Dict[str, Any]
    input: Any
    output: Any
    artifacts: Dict[str, Any]


def _trace_to_dict(trace: Union[_PlanTraceLike, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a PlanTrace or dict into a plain dict, with sane defaults.
    """
    if isinstance(trace, dict):
        d = dict(trace)  # shallow copy
    else:
        # dataclass or attr-like object
        if dataclasses.is_dataclass(trace):
            d = dataclasses.asdict(trace)
        else:
            # Generic getattr extraction of likely fields
            d = {
                "trace_id": getattr(trace, "trace_id", None),
                "role": getattr(trace, "role", None),
                "goal": getattr(trace, "goal", None),
                "status": getattr(trace, "status", None),
                "metadata": getattr(trace, "metadata", {}) or {},
                "input": getattr(trace, "input", None),
                "output": getattr(trace, "output", None),
                "artifacts": getattr(trace, "artifacts", {}) or {},
            }

    # add standard envelope fields if missing
    d.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    d.setdefault("kind", "plan_trace")

    # ensure required keys exist
    d.setdefault("trace_id", f"trace-{int(time.time()*1000)}")
    d.setdefault("role", "system")
    d.setdefault("goal", "")
    d.setdefault("status", "unknown")
    d.setdefault("metadata", {})
    d.setdefault("artifacts", {})

    # ensure metadata has simple run/span helpers if caller wants them
    if not isinstance(d["metadata"], dict):
        d["metadata"] = {"_coerced": True, "value": str(d["metadata"])}

    return d


class TraceLogger:
    """
    Fan-out trace logger with multiple optional sinks:
      - JSONL file (append-only)
      - Memory store (app.state.memory.plan_traces.*)
      - Event publisher (publish(topic, payload))
      - Stdout (for dev)

    Thread-safe, JSON-safe (uses sanitize()) and resilient to NumPy / datetimes.
    """

    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        memory: Optional[Any] = None,
        event_publisher: Optional[Any] = None,
        enable_stdout: bool = False,
        default_topic: str = "plan_trace",
    ):
        self._lock = threading.Lock()
        self._jsonl_path = None  # set via setter to create folder
        if jsonl_path:
            self.set_jsonl_path(jsonl_path)

        self._memory = memory
        self._publisher = event_publisher
        self._enable_stdout = enable_stdout
        self._default_topic = default_topic

        # dedup / last id cache to avoid repeated logs if caller retries
        self._recent_ids = {}
        self._recent_max = 4096

    # -------------------- configuration ---------------------------------

    def set_jsonl_path(self, path: str) -> None:
        with self._lock:
            self._jsonl_path = path
            Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

    def set_memory(self, memory: Any) -> None:
        self._memory = memory

    def set_event_publisher(self, publisher: Any) -> None:
        self._publisher = publisher

    def enable_stdout(self, on: bool = True) -> None:
        self._enable_stdout = on

    def set_default_topic(self, topic: str) -> None:
        self._default_topic = topic

    # -------------------- sinks -----------------------------------------

    def _sink_jsonl(self, record: Dict[str, Any]) -> None:
        if not self._jsonl_path:
            return
        try:
            # sanitize first; then dump once to ensure it's serializable
            line = dumps_safe(record, ensure_ascii=False)
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
        except Exception:
            # never crash the caller
            pass

    def _sink_memory(self, record: Dict[str, Any]) -> None:
        if not self._memory:
            return
        try:
            # Accept a few common repo shapes:
            repo = getattr(self._memory, "plan_traces", None) or getattr(self._memory, "traces", None)
            if repo is None:
                return

            # Try common method names; sanitize first for JSONB
            payload = sanitize(record)
            for method_name in ("insert", "create", "add", "log", "upsert"):
                m = getattr(repo, method_name, None)
                if callable(m):
                    m(payload)
                    return
            # If repo has a generic save() or put():
            for method_name in ("save", "put", "write"):
                m = getattr(repo, method_name, None)
                if callable(m):
                    m(payload)
                    return
        except Exception:
            # don’t kill the main flow on storage error
            pass

    def _sink_publish(self, record: Dict[str, Any], topic: Optional[str]) -> None:
        if not self._publisher:
            return
        try:
            t = topic or self._default_topic
            # accept either .publish(topic, payload) or .emit(topic, payload)
            pub = getattr(self._publisher, "publish", None) or getattr(self._publisher, "emit", None)
            if callable(pub):
                pub(t, sanitize(record))
        except Exception:
            pass

    def _sink_stdout(self, record: Dict[str, Any]) -> None:
        if not self._enable_stdout:
            return
        try:
            sys.stdout.write(dumps_safe(record, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    # -------------------- public API ------------------------------------

    def log(self, trace: Union[_PlanTraceLike, Dict[str, Any]], *, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Log a single PlanTrace (or dict with the same keys).
        Returns the sanitized record actually written.
        """
        with self._lock:
            record = _trace_to_dict(trace)

            # dedupe (simple LRU on trace_id)
            tid = str(record.get("trace_id", ""))
            if tid:
                if tid in self._recent_ids:
                    # update timestamp only; still fan out (idempotent behavior can be changed here)
                    pass
                self._recent_ids[tid] = time.time()
                if len(self._recent_ids) > self._recent_max:
                    # drop oldest ~25% to keep memory bounded
                    for _ in range(self._recent_max // 4):
                        try:
                            self._recent_ids.pop(next(iter(self._recent_ids)))
                        except Exception:
                            break

            # fan out to sinks
            self._sink_jsonl(record)
            self._sink_memory(record)
            self._sink_publish(record, topic)
            self._sink_stdout(record)

            return sanitize(record)

    @contextmanager
    def span(
        self,
        *,
        trace_id: Optional[str] = None,
        role: str = "system",
        goal: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
    ):
        """
        Context manager that logs a start/end pair with duration and error capture.

        Usage:
            with trace_logger.span(role="solver", goal="answer climate query") as log:
                # optional mid-span note
                log.note(status="searching", meta={"depth": 2})
                # your work...
                log.success(output="final answer text")

        The returned object has:
          - note(status: str = "", meta: dict = None, output: Any = None)
          - success(output: Any = None, artifacts: dict = None)
          - error(exc: Exception, artifacts: dict = None)
        """
        start_ts = time.time()
        tid = trace_id or f"span-{int(start_ts*1000)}"
        base_meta = dict(metadata or {})
        # start
        self.log(
            {
                "trace_id": tid,
                "role": role,
                "goal": goal,
                "status": "started",
                "metadata": base_meta,
                "input": None,
                "output": "",
                "artifacts": {},
            },
            topic=topic,
        )

        class _SpanHandle:
            def note(self, status: str = "", meta: Optional[Dict[str, Any]] = None, output: Any = None):
                m = dict(base_meta)
                if meta:
                    m.update(meta)
                self_ = {
                    "trace_id": tid,
                    "role": role,
                    "goal": goal,
                    "status": status or "progress",
                    "metadata": m,
                    "input": None,
                    "output": output,
                    "artifacts": {},
                }
                self_log = self_.copy()
                self_log["metadata"]["t_rel_ms"] = int((time.time() - start_ts) * 1000)
                self.log(self_log, topic=topic)  # type: ignore[attr-defined]

            def success(self, output: Any = None, artifacts: Optional[Dict[str, Any]] = None):
                dur = time.time() - start_ts
                self_log = {
                    "trace_id": tid,
                    "role": role,
                    "goal": goal,
                    "status": "completed",
                    "metadata": {**base_meta, "duration_ms": int(dur * 1000)},
                    "input": None,
                    "output": output,
                    "artifacts": artifacts or {},
                }
                self.log(self_log, topic=topic)  # type: ignore[attr-defined]

            def error(self, exc: BaseException, artifacts: Optional[Dict[str, Any]] = None):
                dur = time.time() - start_ts
                self_log = {
                    "trace_id": tid,
                    "role": role,
                    "goal": goal,
                    "status": "error",
                    "metadata": {**base_meta, "duration_ms": int(dur * 1000), "error": str(exc)},
                    "input": None,
                    "output": "",
                    "artifacts": artifacts or {},
                }
                self.log(self_log, topic=topic)  # type: ignore[attr-defined]

            # allow access to outer logger in inner class
            log = self.log  # type: ignore

        handle = _SpanHandle()
        try:
            yield handle
        except BaseException as e:
            handle.error(e)
            raise
        else:
            handle.success()

# Global instance used across the codebase
trace_logger = TraceLogger(
    jsonl_path="logs/plan_traces.jsonl",
    enable_stdout=False,  # flip to True during local dev
)
