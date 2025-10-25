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
import base64
from typing import Any, AsyncIterator, Callable, Dict, Optional

from .hybrid_bus import HybridKnowledgeBus

log = logging.getLogger("stephanie.bus.nats_client")

class _Msg:
    __slots__ = ("subject", "data")
    def __init__(self, subject: str, data: bytes):
        self.subject = subject
        self.data = data

class _JSFacade:
    """
    Thin async facade over HybridKnowledgeBus that:
    - normalizes publish inputs (dict|bytes)
    - offers both callback-style and async-iterator subscriptions
    """
    def __init__(self, bus: HybridKnowledgeBus):
        self._bus = bus
        self._queues: Dict[str, asyncio.Queue] = {}

    # -------------------- helpers --------------------

    def _make_binary_envelope(
        self, body: bytes, *, content_type: str = "application/octet-stream",
        content_encoding: Optional[str] = None
    ) -> Dict[str, Any]:
        env = {
            "__binary__": True,
            "content_type": content_type,
            "data_b64": base64.b64encode(body).decode("ascii"),
        }
        if content_encoding:
            env["content_encoding"] = content_encoding
        return env

    def _unwrap_binary_envelope(self, payload: Any) -> Optional[bytes]:
        if isinstance(payload, dict) and payload.get("__binary__") and "data_b64" in payload:
            try:
                return base64.b64decode(payload["data_b64"])
            except Exception:
                return None
        return None

    # -------------------- publish / request --------------------

    async def publish(
        self,
        subject: str,
        payload: Any,
        *,
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Publish a message.

        Accepted payload types:
          - dict/list/number -> JSON via bus
          - str              -> utf-8 via bus JSON wrapper {"payload": str}
          - bytes/bytearray/memoryview -> raw bytes (preferred)
              * If the bus does not support raw bytes, a base64 envelope is sent.
        """
        headers = dict(headers or {})

        # raw binary payload
        if isinstance(payload, (bytes, bytearray, memoryview)):
            body = bytes(payload)

            # If HybridKnowledgeBus supports raw publish, use it
            if hasattr(self._bus, "publish_raw"):
                await self._bus.publish_raw(subject, body, headers=headers)
                return

            # Otherwise, send a base64 envelope as JSON
            # (Optionally infer gzip by magic number 1f 8b)
            content_encoding = "gzip" if len(body) >= 2 and body[0] == 0x1F and body[1] == 0x8B else None
            env = self._make_binary_envelope(body, content_encoding=content_encoding)
            await self._bus.publish(subject, env, headers=headers)
            return

        # str -> wrap
        if isinstance(payload, str):
            payload = {"payload": payload}

        # dict/list/number -> pass through (bus JSON-encodes)
        await self._bus.publish(subject, payload, headers=headers)

    async def request(
        self,
        subject: str,
        payload: Dict[str, Any],
        timeout: float = 5.0,
        *,
        allow_binary: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Request/response path. Returns a dict if possible.
        If the remote replies with a binary envelope and allow_binary=True,
        returns {"__binary__": True, "data": <bytes>, ...}.
        """
        resp = await self._bus.request(subject, payload, timeout)
        if resp is None:
            return None

        # If bus returns bytes, try JSON, else envelope
        if isinstance(resp, (bytes, bytearray, memoryview)):
            raw = bytes(resp)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {"__binary__": True, "data": raw} if allow_binary else None

        # If bus returns dict, maybe it’s a binary envelope
        if isinstance(resp, dict):
            raw = self._unwrap_binary_envelope(resp)
            if raw is not None and allow_binary:
                return {"__binary__": True, "data": raw, **{k: v for k, v in resp.items() if k not in ("__binary__", "data_b64")}}
            return resp

        # Anything else is unexpected
        return None

    # -------------------- subscribe --------------------

    async def subscribe(
        self,
        subject: str,
        handler: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> AsyncIterator[_Msg] | None:
        """
        Two modes:
          1) handler provided  -> callback receives dict payloads (JSON-decoded)
          2) no handler        -> returns async iterator yielding _Msg with raw bytes
             - If the bus delivers dicts, we JSON-encode them to bytes.
             - If the bus delivers a base64 binary envelope, we decode to bytes.
        """
        if handler is not None:
            # Pass-through for dict-typed callbacks
            await self._bus.subscribe(subject, handler)
            return None

        q: asyncio.Queue = self._queues.get(subject) or asyncio.Queue(maxsize=1024)
        self._queues[subject] = q

        async def _enqueue(payload: Any):
            # 1) Raw bytes from bus -> pass through
            if isinstance(payload, (bytes, bytearray, memoryview)):
                data = bytes(payload)
            else:
                # 2) Binary envelope?
                raw = self._unwrap_binary_envelope(payload)
                if raw is not None:
                    data = raw
                else:
                    # 3) Dict -> JSON bytes; str -> utf-8
                    try:
                        if isinstance(payload, str):
                            data = payload.encode("utf-8")
                        else:
                            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    except Exception:
                        data = b"{}"

            try:
                q.put_nowait(_Msg(subject, data))
            except asyncio.QueueFull:
                # Drop oldest to make room (lossy but safe for telemetry)
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                q.put_nowait(_Msg(subject, data))

        await self._bus.subscribe(subject, _enqueue)

        async def _aiter():
            while True:
                msg = await q.get()
                yield msg

        class _AIter:
            def __aiter__(self):
                return _aiter()

        return _AIter()

    # -------------------- ops passthrough --------------------

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
