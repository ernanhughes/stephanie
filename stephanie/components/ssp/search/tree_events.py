from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Callable

from stephanie.utils.json_sanitize import sanitize
from stephanie.components.ssp.util import get_trace_logger, PlanTrace_safe


def _node_rec(n: Any) -> Dict[str, Any]:
    """
    Best-effort serialization for a SolutionNode (or similar).
    Works even if some attributes are missing.
    """
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
    Lightweight event bridge for Agentic Tree Search.

    Features:
    - Generic .emit(event, **payload) + convenience methods (on_* names)
    - Any unknown method name is treated as an event (via __getattr__)
    - Fans out to:
        • trace_logger (PlanTrace-like records)
        • optional event publisher (publish/emit)
        • optional memory store (plan_traces repo)
    - Thread-safe; keeps a small recent-events ring buffer for UIs
    - Optional throttle to avoid spamming sinks

    Usage:
        emitter = TreeEventEmitter(topic="ssp.tree")
        emitter.on_root_created(node)
        emitter.on_node_added(parent=parent, child=child)
        emitter.emit("rollout_complete", report=report)
    """

    def __init__(
        self,
        *,
        topic: str = "ssp.tree",
        publisher: Optional[Any] = None,
        memory: Optional[Any] = None,
        throttle_ms: int = 0,
        keep_recent: int = 256,
        enable_trace_log: bool = True,
    ):
        self._topic = str(topic)
        self._publisher = publisher  # should expose publish(topic, payload) or emit(topic, payload)
        self._memory = memory        # optional: if you want to additionally persist
        self._throttle_ms = int(throttle_ms)
        self._last_sent_ts: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._recent = deque(maxlen=int(keep_recent))
        self._trace_logger = get_trace_logger() if enable_trace_log else None

    # ------------------------- public API -------------------------

    def set_publisher(self, publisher: Any) -> None:
        with self._lock:
            self._publisher = publisher

    def set_memory(self, memory: Any) -> None:
        with self._lock:
            self._memory = memory

    def recent(self, limit: int = 50) -> list[Dict[str, Any]]:
        with self._lock:
            return list(list(self._recent)[-int(limit):])

    def emit(self, event: str, **payload: Any) -> None:
        """
        Main event entrypoint. JSON-safe & thread-safe.
        """
        now = time.time()
        with self._lock:
            # throttle identical event name if configured
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

            # keep in-memory ring
            self._recent.append(rec)

        # fan-out without holding the lock
        self._to_trace_logger(event, rec)
        self._to_publisher(event, rec)
        self._to_memory(rec)

    # -------------------- convenience helpers --------------------

    def on_root_created(self, node: Any) -> None:
        self.emit("root_created", node=_node_rec(node))

    def on_node_added(self, parent: Any, child: Any) -> None:
        self.emit("node_added", parent=_node_rec(parent), child=_node_rec(child))

    def on_expand(self, node: Any) -> None:
        self.emit("expand", node=_node_rec(node))

    def on_debug(self, node: Any) -> None:
        self.emit("debug", node=_node_rec(node))

    def on_backprop(self, node: Any, delta: Optional[float] = None) -> None:
        self.emit("backprop", node=_node_rec(node), delta=delta)

    def on_best_update(self, node: Any) -> None:
        self.emit("best_update", node=_node_rec(node))

    def on_progress(self, info: Dict[str, Any]) -> None:
        self.emit("progress", **sanitize(info))

    def on_rollout_complete(self, report: Dict[str, Any]) -> None:
        self.emit("rollout_complete", report=sanitize(report))

    def on_error(self, error: str, where: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        self.emit("error", error=str(error), where=where, extra=sanitize(extra or {}))

    # If the tree calls any unknown method name (e.g., on_prune), treat it as an event:
    def __getattr__(self, name: str) -> Callable[..., None]:
        if name.startswith("on_"):
            event = name[3:]  # strip "on_"
            def _f(**kw: Any) -> None:
                self.emit(event, **kw)
            return _f
        raise AttributeError(name)

    # -------------------------- sinks ----------------------------

    def _to_trace_logger(self, event: str, rec: Dict[str, Any]) -> None:
        if not self._trace_logger:
            return
        try:
            # pack as a PlanTrace-like record for uniform observability
            self._trace_logger.log(PlanTrace_safe(
                trace_id=f"tree-{event}-{int(rec['ts']*1000)%1_000_000}",
                role="tree",
                goal=event,
                status="event",
                metadata={"topic": self._topic},
                input=None,
                output="",
                artifacts=rec,
            ))
        except Exception:
            pass  # never break the caller

    def _to_publisher(self, event: str, rec: Dict[str, Any]) -> None:
        pub = self._publisher
        if not pub:
            return
        try:
            fn = getattr(pub, "publish", None) or getattr(pub, "emit", None)
            if callable(fn):
                fn(f"{self._topic}.{event}", rec)
        except Exception:
            pass

    def _to_memory(self, rec: Dict[str, Any]) -> None:
        if not self._memory:
            return
        try:
            repo = getattr(self._memory, "plan_traces", None) or getattr(self._memory, "traces", None)
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
