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

from stephanie.utils.json_sanitize import dumps_safe, sanitize

# If you have a canonical PlanTrace dataclass you can keep this import;
# the logger works fine even without it (structural typing below).
try:
    from stephanie.data.plan_trace import PlanTrace  # noqa: F401
except Exception:
    PlanTrace = None  # type: ignore


class _PlanTraceLike(Protocol):
    trace_id: str
    role: str
    goal: str
    status: str
    metadata: Dict[str, Any]
    input: Any
    output: Any
    artifacts: Dict[str, Any]


def _trace_to_dict(trace: Union[_PlanTraceLike, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(trace, dict):
        d = dict(trace)
    else:
        if dataclasses.is_dataclass(trace):
            d = dataclasses.asdict(trace)
        else:
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

    d.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    d.setdefault("kind", "plan_trace")
    d.setdefault("trace_id", f"trace-{int(time.time()*1000)}")
    d.setdefault("role", "system")
    d.setdefault("goal", "")
    d.setdefault("status", "unknown")
    d.setdefault("metadata", {})
    d.setdefault("artifacts", {})

    if not isinstance(d["metadata"], dict):
        d["metadata"] = {"_coerced": True, "value": str(d["metadata"])}

    return d


class TraceLogger:
    """Fan-out trace logger (JSONL file, memory repo, publisher, stdout). Thread-safe."""

    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        memory: Optional[Any] = None,
        event_publisher: Optional[Any] = None,
        enable_stdout: bool = False,
        default_topic: str = "plan_trace",
    ):
        self._lock = threading.Lock()
        self._jsonl_path = None
        if jsonl_path:
            self.set_jsonl_path(jsonl_path)

        self._memory = memory
        self._publisher = event_publisher
        self._enable_stdout = enable_stdout
        self._default_topic = default_topic

        self._recent_ids = {}
        self._recent_max = 4096

    # ---------- config ----------
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

    # ---------- sinks ----------
    def _sink_jsonl(self, record: Dict[str, Any]) -> None:
        if not self._jsonl_path:
            return
        try:
            line = dumps_safe(record, ensure_ascii=False)
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _sink_memory(self, record: Dict[str, Any]) -> None:
        if not self._memory:
            return
        try:
            repo = getattr(self._memory, "plan_traces", None) or getattr(self._memory, "traces", None)
            if repo is None:
                return
            payload = sanitize(record)
            for mname in ("insert", "create", "add", "log", "upsert"):
                m = getattr(repo, mname, None)
                if callable(m):
                    m(payload)
                    return
            for mname in ("save", "put", "write"):
                m = getattr(repo, mname, None)
                if callable(m):
                    m(payload)
                    return
        except Exception:
            pass

    def _sink_publish(self, record: Dict[str, Any], topic: Optional[str]) -> None:
        if not self._publisher:
            return
        try:
            t = topic or self._default_topic
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

    # ---------- public API ----------
    def log(self, trace: Union[_PlanTraceLike, Dict[str, Any]], *, topic: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            record = _trace_to_dict(trace)

            tid = str(record.get("trace_id", ""))
            if tid:
                self._recent_ids[tid] = time.time()
                if len(self._recent_ids) > self._recent_max:
                    for _ in range(self._recent_max // 4):
                        try:
                            self._recent_ids.pop(next(iter(self._recent_ids)))
                        except Exception:
                            break

            self._sink_jsonl(record)
            self._sink_memory(record)
            self._sink_publish(record, topic)
            self._sink_stdout(record)
            return sanitize(record)

    def emit(
        self,
        tag: str,
        *,
        role: str = "system",
        goal: str = "",
        status: str = "info",
        meta: Optional[Dict[str, Any]] = None,
        output: Any = "",
        artifacts: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience for quick trace lines (used widely in SSP/Jitter code).
        """
        rec = {
            "trace_id": trace_id or f"{tag}-{int(time.time()*1000)}",
            "role": role,
            "goal": goal,
            "status": status,
            "metadata": {"tag": tag, **(meta or {})},
            "input": None,
            "output": output,
            "artifacts": artifacts or {},
        }
        return self.log(rec, topic=topic)

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
        start_ts = time.time()
        tid = trace_id or f"span-{int(start_ts*1000)}"
        base_meta = dict(metadata or {})
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

        logger = self

        class _SpanHandle:
            def note(self, status: str = "", meta: Optional[Dict[str, Any]] = None, output: Any = None):
                m = dict(base_meta)
                if meta:
                    m.update(meta)
                logger.log(
                    {
                        "trace_id": tid,
                        "role": role,
                        "goal": goal,
                        "status": status or "progress",
                        "metadata": {**m, "t_rel_ms": int((time.time() - start_ts) * 1000)},
                        "input": None,
                        "output": output,
                        "artifacts": {},
                    },
                    topic=topic,
                )

            def success(self, output: Any = None, artifacts: Optional[Dict[str, Any]] = None):
                dur = time.time() - start_ts
                logger.log(
                    {
                        "trace_id": tid,
                        "role": role,
                        "goal": goal,
                        "status": "completed",
                        "metadata": {**base_meta, "duration_ms": int(dur * 1000)},
                        "input": None,
                        "output": output,
                        "artifacts": artifacts or {},
                    },
                    topic=topic,
                )

            def error(self, exc: BaseException, artifacts: Optional[Dict[str, Any]] = None):
                dur = time.time() - start_ts
                logger.log(
                    {
                        "trace_id": tid,
                        "role": role,
                        "goal": goal,
                        "status": "error",
                        "metadata": {**base_meta, "duration_ms": int(dur * 1000), "error": str(exc)},
                        "input": None,
                        "output": "",
                        "artifacts": artifacts or {},
                    },
                    topic=topic,
                )

        handle = _SpanHandle()
        try:
            yield handle
        except BaseException as e:
            handle.error(e)
            raise
        else:
            handle.success()


# Module-global default (safe to keep)
trace_logger = TraceLogger(
    jsonl_path="logs/plan_traces.jsonl",
    enable_stdout=False,
)

# --------- NEW: singleton accessors / SIS binding ---------
_GLOBAL_LOGGER: Optional[TraceLogger] = trace_logger  # lift the module instance as the singleton


def get_trace_logger(
    *,
    jsonl_path: Optional[str] = None,
    memory: Optional[Any] = None,
    event_publisher: Optional[Any] = None,
    enable_stdout: Optional[bool] = None,
    default_topic: Optional[str] = None,
) -> TraceLogger:
    """
    Return a process-wide TraceLogger singleton. Optionally (re)configure sinks.
    """
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = TraceLogger()

    if jsonl_path:
        _GLOBAL_LOGGER.set_jsonl_path(jsonl_path)
    if memory is not None:
        _GLOBAL_LOGGER.set_memory(memory)
    if event_publisher is not None:
        _GLOBAL_LOGGER.set_event_publisher(event_publisher)
    if enable_stdout is not None:
        _GLOBAL_LOGGER.enable_stdout(enable_stdout)
    if default_topic is not None:
        _GLOBAL_LOGGER.set_default_topic(default_topic)

    return _GLOBAL_LOGGER


def attach_to_app(app, *, jsonl_path: Optional[str] = None, enable_stdout: Optional[bool] = None) -> TraceLogger:
    """
    Bind the singleton to SIS app state (memory + publisher), and stash it on app.state.
    """
    tl = get_trace_logger(
        jsonl_path=jsonl_path,
        enable_stdout=enable_stdout,
        memory=getattr(app.state, "memory", None),
        event_publisher=getattr(app.state, "event_publisher", None),
    )
    setattr(app.state, "trace_logger", tl)
    return tl


__all__ = ["TraceLogger", "trace_logger", "get_trace_logger", "attach_to_app"]
