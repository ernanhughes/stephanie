# stephanie/components/tree/events.py
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Protocol

from stephanie.utils.json_sanitize import sanitize


class Publisher(Protocol):
    def publish(self, topic: str, payload: Dict[str, Any]) -> None: ...
    # or alternative signature:
    def emit(self, topic: str, payload: Dict[str, Any]) -> None: ...


class TraceRepo(Protocol):
    def insert(self, payload: Dict[str, Any]) -> None: ...
    # compatible fallbacks: add/create/log/save/put/write


def _node_rec(n: Any) -> Dict[str, Any]:
    """Best-effort serialization for a SolutionNode (or similar)."""
    if n is None:
        return {}
    return sanitize({
        "id": getattr(n, "id", None),
        "parent_id": getattr(n, "parent_id", None),
        "root_id": getattr(n, "root_id", None),
        "depth": getattr(n, "depth", None),
        "sibling_index": getattr(n, "sibling_index", None),
        "node_type": getattr(n, "node_type", None),
        "plan": getattr(n, "plan", None),
        "summary": getattr(n, "summary", None),
        "metric": getattr(n, "metric", None),
        "is_buggy": getattr(n, "is_buggy", None),
        "task_description": getattr(n, "task_description", None),
    })


class TreeEventEmitter:
    """
    Lightweight, reusable event bridge for Agentic Tree Search.

    - Async .emit(event, **payload) with sync fallback (.emit_sync)
    - Fire-and-forget on_* convenience methods (safe in sync or async code)
    - Optional sinks: publisher (publish/emit), memory store, tracer callback
    - Thread-safe ring buffer of recent events for UIs
    - Optional throttle per-event to prevent spam
    """

    def __init__(
        self,
        *,
        topic: str = "tree",
        publisher: Optional[Publisher] = None,
        memory: Optional[Any] = None,
        tracer: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        throttle_ms: int = 0,
        keep_recent: int = 256,
    ):
        self._topic = str(topic)
        self._publisher = publisher
        self._memory = memory          # expects a repo-like object under .plan_traces or .traces
        self._tracer = tracer          # callable(event, record_dict)
        self._throttle_ms = int(throttle_ms)

        self._last_sent_ts: Dict[str, float] = {}
        self._recent = deque(maxlen=int(keep_recent))
        self._lock = threading.Lock()

    # ------------------------- public API -------------------------

    def set_publisher(self, publisher: Publisher) -> None:
        with self._lock:
            self._publisher = publisher

    def set_memory(self, memory: Any) -> None:
        with self._lock:
            self._memory = memory

    def set_tracer(self, tracer: Optional[Callable[[str, Dict[str, Any]], None]]) -> None:
        with self._lock:
            self._tracer = tracer

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(list(self._recent)[-int(limit):])

    async def emit(self, event: str, **payload: Any) -> None:
        """Awaitable event entrypoint with throttling and thread safety."""
        now = time.time()
        with self._lock:
            if self._throttle_ms > 0:
                last = self._last_sent_ts.get(event, 0.0)
                if (now - last) * 1000.0 < self._throttle_ms:
                    return
                self._last_sent_ts[event] = now

            rec = sanitize({
                "kind": "tree_event",
                "ts": now,
                "event": event,
                "topic": self._topic,
                "payload": payload,
            })
            self._recent.append(rec)

        self._to_tracer(event, rec)
        self._to_publisher(event, rec)
        self._to_memory(rec)

    def emit_sync(self, event: str, **payload: Any) -> None:
        """Synchronous fallback when no event loop is running."""
        now = time.time()
        with self._lock:
            if self._throttle_ms > 0:
                last = self._last_sent_ts.get(event, 0.0)
                if (now - last) * 1000.0 < self._throttle_ms:
                    return
                self._last_sent_ts[event] = now

            rec = sanitize({
                "kind": "tree_event",
                "ts": now,
                "event": event,
                "topic": self._topic,
                "payload": payload,
            })
            self._recent.append(rec)

        self._to_tracer(event, rec)
        self._to_publisher(event, rec)
        self._to_memory(rec)

    # -------------------- convenience helpers --------------------

    def _fire(self, event: str, **kw: Any) -> None:
        """Fire-and-forget that works both in and outside an event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event, **kw))
        except RuntimeError:
            self.emit_sync(event, **kw)

    # Canonical tree events (add more as needed)
    def on_root_created(self, node: Any) -> None:
        self._fire("root_created", node=_node_rec(node))

    def on_node_added(self, parent: Any, child: Any) -> None:
        self._fire("node_added", parent=_node_rec(parent), child=_node_rec(child))

    def on_expand(self, node: Any) -> None:
        self._fire("expand", node=_node_rec(node))

    def on_debug(self, node: Any) -> None:
        self._fire("debug", node=_node_rec(node))

    def on_backprop(self, node: Any, delta: Optional[float] = None) -> None:
        self._fire("backprop", node=_node_rec(node), delta=delta)

    def on_best_update(self, node: Any) -> None:
        self._fire("best_update", node=_node_rec(node))

    def on_progress(self, info: Dict[str, Any]) -> None:
        self._fire("progress", **sanitize(info))

    def on_rollout_complete(self, report: Dict[str, Any]) -> None:
        self._fire("rollout_complete", report=sanitize(report))

    def on_error(self, error: str, where: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        self._fire("error", error=str(error), where=where, extra=sanitize(extra or {}))

    # Treat any unknown on_* as a generic event name
    def __getattr__(self, name: str) -> Callable[..., None]:
        if name.startswith("on_"):
            event = name[3:]
            def _f(**kw: Any) -> None:
                self._fire(event, **kw)
            return _f
        raise AttributeError(name)

    # -------------------------- sinks ----------------------------

    def _to_tracer(self, event: str, rec: Dict[str, Any]) -> None:
        tracer = self._tracer
        if not tracer:
            return
        try:
            tracer(event, rec)
        except Exception:
            pass

    def _to_publisher(self, event: str, rec: Dict[str, Any]) -> None:
        pub = self._publisher
        if not pub:
            return
        try:
            if hasattr(pub, "publish") and callable(pub.publish):
                pub.publish(f"{self._topic}.{event}", rec)
            elif hasattr(pub, "emit") and callable(pub.emit):
                pub.emit(f"{self._topic}.{event}", rec)
        except Exception:
            pass

    def _to_memory(self, rec: Dict[str, Any]) -> None:
        mem = self._memory
        if not mem:
            return
        try:
            repo = getattr(mem, "plan_traces", None) or getattr(mem, "traces", None)
            if not repo:
                return
            payload = sanitize({
                "trace_id": f"tree-{int(rec['ts']*1000)}",
                "role": "tree",
                "goal": rec.get("event", ""),
                "status": "event",
                "metadata": {"topic": self._topic},
                "input": None,
                "output": "",
                "artifacts": rec,
            })
            for name in ("insert", "add", "create", "log", "save", "put", "write"):
                m = getattr(repo, name, None)
                if callable(m):
                    m(payload)
                    return
        except Exception:
            pass


__all__ = ["TreeEventEmitter", "_node_rec", "Publisher", "TraceRepo"]
