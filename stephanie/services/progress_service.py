# stephanie/services/progress_service.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

from stephanie.services.service_protocol import Service


@dataclass
class ProgressTask:
    name: str
    total: int
    started_at: float = field(default_factory=time.time)
    done: int = 0
    last_bucket: int = -1
    last_emit_ts: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running | ok | error | cancelled

class ProgressService(Service):
    def __init__(self, cfg: Dict, logger):
        self.cfg = cfg
        self.logger = logger
        self.service_name = (self.cfg or {}).get("name", "progress")

    def initialize(self, **kwargs) -> None:
        self.cfg = kwargs.get("cfg", {}) or {}
        self.logger = kwargs.get("logger", self.logger)
        self.service_name = self.cfg.get("name", "progress-service-v1")

        self.default_every = int(self.cfg.get("every_percent", 5))
        self.min_interval = float(self.cfg.get("min_interval_sec", 0.4))

        self.console_echo   = bool(self.cfg.get("console_echo", True))
        self.console_prefix = str(self.cfg.get("console_prefix", "[progress]"))

        self._tasks: Dict[str, ProgressTask] = {}
        # Either make it re-entrant...
        self._lock = threading.RLock()
        # ...and still avoid calling _emit while holding the lock.

    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            active = {
                k: dict(
                    name=v.name, total=v.total, done=v.done, status=v.status,
                    started_at=v.started_at
                )
                for k, v in self._tasks.items() if v.status == "running"
            }
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {"active_tasks": len(active)},
            "active": active,
        }

    def shutdown(self) -> None:
        pass

    @property
    def name(self) -> str:
        return self.service_name

    # ---- Core API ----
    def start(self, task: str, total: int, *, meta: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._tasks[task] = ProgressTask(name=task, total=max(1, int(total)), meta=meta or {})
        self._emit(task, substage="start", done=0, force=True)

    def set(self, task: str, done: int, *, total: Optional[int] = None,
            substage: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            t = self._tasks.get(task)
            if not t:
                t = ProgressTask(name=task, total=max(1, int(total or 1)))
                self._tasks[task] = t
            if total is not None:
                t.total = max(1, int(total))
            t.done = max(0, min(int(done), t.total))
        self._maybe_emit(task, substage=substage, extra=extra)

    def tick(self, task: str, n: int = 1, *, substage: Optional[str] = None,
             extra: Optional[Dict[str, Any]] = None, **kw) -> None:
        """
        Backward-compatible: supports either increments (n)
        or absolute updates via tick(task, done=..., total=...).
        """
        if "done" in kw:
            self.set(task, int(kw["done"]), total=kw.get("total"), substage=substage, extra=extra)
            return

        with self._lock:
            t = self._tasks.get(task)
            if not t:
                t = ProgressTask(name=task, total=1)
                self._tasks[task] = t
            t.done = max(0, min(t.done + int(n), t.total))
        self._maybe_emit(task, substage=substage, extra=extra)

    def end(self, task: str, status: str = "ok", *, extra: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            t = self._tasks.get(task)
            if t:
                t.done = t.total
                t.status = status
        self._emit(task, substage="end", force=True, extra=extra)

    # Alias for older call sites
    def done(self, task: str, **kw) -> None:
        self.end(task, **kw)

    def stage(self, task: str, stage: str, **kw) -> None:
        with self._lock:
            if task not in self._tasks:
                self._tasks[task] = ProgressTask(name=task, total=1)
        self._emit(task, substage=stage, extra=kw, force=True)

    def mk_cb(self, *, task: str, every: Optional[int] = None) -> Callable[..., None]:
        step = max(1, int(every or self.default_every))
        with self._lock:
            t = self._tasks.get(task) or ProgressTask(name=task, total=1)
            t.meta["every_percent"] = step
            self._tasks[task] = t

        def _cb(*args):
            if not args:
                return
            if isinstance(args[0], str):
                substage = args[0]
                done  = int(args[1]) if len(args) > 1 and args[1] is not None else 0
                total = int(args[2]) if len(args) > 2 and args[2] is not None else 1
                extra = args[3] if len(args) > 3 else None
                self.set(task, done, total=total, substage=substage, extra=extra)
            else:
                done  = int(args[0]) if len(args) > 0 and args[0] is not None else 0
                total = int(args[1]) if len(args) > 1 and args[1] is not None else 1
                extra = args[2] if len(args) > 2 else None
                self.set(task, done, total=total, extra=extra)
        return _cb

    def tqdm(self, iterable: Iterable, *, task: str, total: Optional[int] = None,
             substage: Optional[str] = None):
        if total is None:
            try: total = len(iterable)
            except Exception: total = 0
        self.start(task, total or 1)
        for i, item in enumerate(iterable, 1):
            yield item
            self.set(task, i, total=total or max(i, 1), substage=substage)
        self.end(task, "ok")

    def scope(self, task: str, total: int, *, meta: Optional[Dict[str, Any]] = None):
        ps = self
        class _Scope:
            def __enter__(self_inner):
                ps.start(task, total, meta=meta)
                return ps
            def __exit__(self_inner, exc_type, exc, tb):
                ps.end(task, "error" if exc else "ok")
        return _Scope()

    # ---- Internals ----
    def _maybe_emit(self, task: str, *, substage: Optional[str], extra: Optional[Dict[str, Any]]) -> None:
        # Decide under the lock...
        with self._lock:
            t = self._tasks.get(task)
            if not t:
                return
            total = max(1, t.total)
            pct = int((t.done / total) * 100)
            step = int(t.meta.get("every_percent", self.default_every))
            bucket = pct // max(1, step)
            now = time.time()
            should_emit = (bucket != t.last_bucket) or ((now - t.last_emit_ts) >= self.min_interval) or (t.done == total)
            if should_emit:
                t.last_bucket = bucket
                t.last_emit_ts = now

        # ...then emit OUTSIDE the lock (prevents deadlocks)
        if should_emit:
            self._emit(task, substage=substage, extra=extra)

    def _emit(self, task: str, *, substage: Optional[str] = None,
              done: Optional[int] = None, force: bool = False,
              extra: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            t = self._tasks.get(task)
            if not t:
                return
            total = max(1, t.total)
            d = t.done if done is None else done
            pct = int((d / total) * 100)
            payload = {
                "stage": task,
                "done": int(d),
                "total": int(total),
                "percent": pct,
                "status": t.status,
                **(t.meta or {}),
            }
            if substage:
                payload["substage"] = substage
            if extra:
                payload.update(extra)

        try:
            if self.logger:
                self.logger.log("Progress", payload)
        except Exception:
            pass

        if self.console_echo and t.meta.get("console_echo", True):
            try:
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"{self.console_prefix} {task}: {pct:3d}% ({int(d)}/{int(total)})"
                            + (f" | {payload.get('substage')}" if payload.get("substage") else ""))
            except Exception:
                print(f"{self.console_prefix} {task}: {pct:3d}% ({int(d)}/{int(total)})"
                    + (f" | {payload.get('substage')}" if payload.get("substage") else ""),
                    flush=True)

