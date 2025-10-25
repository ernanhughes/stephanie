"""
Unified JetStream client facade for Stephanie.

- Prefers NATS via HybridKnowledgeBus; falls back to in-proc when allowed.
- Presents a tiny NATS-like interface expected by JAS and dashboards:
    await js.publish(subject, payload)        # payload: dict OR bytes
    await js.subscribe(subject)               # -> async iterator yielding Msg(data, subject)
    await js.subscribe(subject, handler=cb)   # handler(payload_dict)
    await js.request(subject, payload_dict, timeout=5.0) -> dict|None
    await js.flush(timeout=1.0)
    await js.drain_subject(subject)
    await js.close()
    js.health_check()

Env overrides (optional):
  STEPH_BUS_ENABLED=1|0
  STEPH_BUS_BACKEND=nats|inproc
  STEPH_BUS_SERVERS=nats://localhost:4222,nats://other:4222
  STEPH_BUS_STREAM=stephanie
  STEPH_BUS_REQUIRED=0|1
  STEPH_BUS_FALLBACK=inproc|none
  STEPH_BUS_CONNECT_TIMEOUT=2.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Callable, Dict, Optional

from .hybrid_bus import HybridKnowledgeBus

log = logging.getLogger("stephanie.bus.nats_client")

# --------------------------- facade message ---------------------------

class _Msg:
    """Lightweight NATS-like message for async-iterator subscriptions."""
    __slots__ = ("subject", "data")
    def __init__(self, subject: str, data: bytes):
        self.subject = subject
        self.data = data  # bytes


# --------------------------- facade client ----------------------------

class _JSFacade:
    """
    Thin async facade over HybridKnowledgeBus that:
    - normalizes publish inputs (dict|bytes)
    - offers both callback-style and async-iterator subscriptions
    """
    def __init__(self, bus: HybridKnowledgeBus):
        self._bus = bus
        self._queues: Dict[str, asyncio.Queue] = {}  # subject -> queue for iter subs

    # ---- publish / request ----

    async def publish(self, subject: str, payload: Any) -> None:
        """
        Accepts dict or bytes. We always hand dict to the bus (it JSON-encodes).
        """
        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = json.loads(payload.decode("utf-8"))
            except Exception:
                # as a last resort, wrap raw bytes into envelope
                payload = {"payload_raw": True, "data": payload.decode("utf-8", "ignore")}
        elif not isinstance(payload, dict):
            # allow simple types: wrap them
            payload = {"payload": payload}
        await self._bus.publish(subject, payload)

    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        return await self._bus.request(subject, payload, timeout)

    # ---- subscribe ----

    async def subscribe(self, subject: str, handler: Optional[Callable[[Dict[str, Any]], Any]] = None) -> AsyncIterator[_Msg] | None:
        """
        Two modes:
          1) handler provided  -> returns None (callback receives payload dict)
          2) no handler        -> returns async iterator yielding _Msg with .data bytes
        """
        if handler is not None:
            # Pass-through callback style (payload dict)
            await self._bus.subscribe(subject, handler)
            return None

        # Async-iterator mode: build a queue + handler that enqueues bytes
        q: asyncio.Queue = self._queues.get(subject) or asyncio.Queue(maxsize=1024)
        self._queues[subject] = q

        async def _enqueue(payload: Dict[str, Any]):
            try:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            except Exception:
                # ensure we never break the consumer
                data = b"{}"
            try:
                q.put_nowait(_Msg(subject, data))
            except asyncio.QueueFull:
                # Drop oldest to make room (lossy but safe for telemetry)
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                with contextless(q):
                    q.put_nowait(_Msg(subject, data))

        await self._bus.subscribe(subject, _enqueue)

        # Return an async iterator over the queue
        async def _aiter():
            while True:
                msg = await q.get()
                yield msg

        class _AIter:
            def __aiter__(self):
                return _aiter()

        return _AIter()

    # ---- ops passthrough ----

    async def flush(self, timeout: float = 1.0) -> bool:
        if hasattr(self._bus, "flush"):
            return await self._bus.flush(timeout=timeout)
        return False

    async def drain_subject(self, subject: str) -> bool:
        if hasattr(self._bus, "drain_subject"):
            return await self._bus.drain_subject(subject)
        return False

    async def close(self) -> None:
        await self._bus.close()

    # ---- health ----

    def health_check(self) -> Dict[str, Any]:
        if hasattr(self._bus, "health_check"):
            return self._bus.health_check()
        return {"is_healthy": False, "details": "no health_check available"}


# --------------------------- singleton loader --------------------------

_JS_SINGLETON: Optional[_JSFacade] = None
_JS_LOCK = asyncio.Lock()

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _env_list(name: str, default: list) -> list:
    raw = os.getenv(name)
    if not raw:
        return default
    return [p.strip() for p in raw.split(",") if p.strip()]

async def get_js(cfg: Optional[Dict[str, Any]] = None) -> _JSFacade:
    """
    Return a process-wide singleton facade connected via HybridKnowledgeBus.
    Safe to call many times; the first call establishes the connection.
    """
    global _JS_SINGLETON
    if _JS_SINGLETON is not None:
        return _JS_SINGLETON

    async with _JS_LOCK:
        if _JS_SINGLETON is not None:
            return _JS_SINGLETON

        # Build effective config (env -> overrides -> defaults)
        eff = dict(cfg or {})
        eff.setdefault("enabled", _env_bool("STEPH_BUS_ENABLED", True))
        eff.setdefault("backend", os.getenv("STEPH_BUS_BACKEND", "nats"))
        eff.setdefault("servers", _env_list("STEPH_BUS_SERVERS", ["nats://localhost:4222"]))
        eff.setdefault("stream", os.getenv("STEPH_BUS_STREAM", "stephanie"))
        eff.setdefault("required", _env_bool("STEPH_BUS_REQUIRED", False))
        eff.setdefault("strict", eff["required"])
        eff.setdefault("connect_timeout_s", _env_float("STEPH_BUS_CONNECT_TIMEOUT", 2.0))
        eff.setdefault("fallback", os.getenv("STEPH_BUS_FALLBACK", "inproc"))

        bus = HybridKnowledgeBus(eff, logger=log)

        ok = await bus.connect(timeout=eff.get("connect_timeout_s"))
        if not ok and eff.get("required", False):
            raise RuntimeError("JetStream bus required but unavailable")

        _JS_SINGLETON = _JSFacade(bus)
        return _JS_SINGLETON


# --------------------------- tiny context helper -----------------------

class contextless:
    """No-op context manager to keep 'with contextless(q):' patterns safe in any env."""
    def __init__(self, *_a, **_kw): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
