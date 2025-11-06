# stephanie/services/cache_service.py
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from stephanie.services.service_protocol import Service

# Soft dependency: nats is only imported when we actually need it.
_nats = None
_js_api = None


def _json_bytes(obj: Any) -> bytes:
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8")
    # canonical JSON
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stable_key(subject: str, payload: Any, *, version: str = "v1") -> str:
    h = hashlib.blake2s(digest_size=16)
    h.update(subject.encode("utf-8"))
    h.update(b"\x00")
    h.update(_json_bytes(payload))
    h.update(b"\x00")
    h.update(version.encode("utf-8"))
    return h.hexdigest()


@dataclass
class CacheConfig:
    bucket: str = "cache2h"
    blobs_bucket: str = "cache_blobs"
    ttl_seconds: int = 2 * 60 * 60
    inline_max: int = 750_000  # bytes; under NATS default max payload
    connect_if_missing: bool = True  # create our own NATS conn if bus doesn't expose JS
    # optional direct NATS settings when we must self-connect
    nats_servers: Any = ("nats://localhost:4222",)


class CacheService(Service):
    """
    JetStream-backed cache service (KV + ObjectStore) with TTL and singleflight.
    - Works with your injected HybridKnowledgeBus when it exposes NATS/JetStream.
    - Falls back to self-connecting to NATS if needed (configurable).
    """

    def __init__(self, cfg: Dict, memory: Any, logger: Any):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self._ready = False
        self._js = None
        self._kv = None
        self._objs = None
        self._nc = None           # only if we self-connect
        self._own_conn = False    # whether we own the NATS connection
        self._bus = None
        self._conf = CacheConfig(**(self.cfg.get("cache", {}) or {}))
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "locks": 0,
            "waits": 0,
        }

    @property
    def name(self) -> str:
        return "cache-bus-v1"

    # -------- Service lifecycle --------

    def set_bus(self, bus: Any) -> None:
        self._bus = bus
        super().set_bus(bus)

    def initialize(self, **kwargs) -> None:
        """
        Synchronous; we lazily setup JetStream on first use to avoid `await` here.
        kwargs can override CacheConfig at registration time.
        """
        if kwargs:
            # Allow per-environment overrides from ServiceContainer.register(..., init_args={...})
            self._conf = CacheConfig(**{**self._conf.__dict__, **kwargs})
        # lazy init done in _ensure_ready()

    def shutdown(self) -> None:
        # Best effort: close self-owned NATS connection
        if self._own_conn and self._nc:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._nc.drain())
                else:
                    loop.run_until_complete(self._nc.drain())
            except Exception:
                pass
        self._ready = False
        self._js = self._kv = self._objs = self._nc = None

    def health_check(self) -> Dict[str, Any]:
        status = "healthy" if self._ready else "initializing"
        return {
            "status": status,
            "bucket": self._conf.bucket,
            "blobs_bucket": self._conf.blobs_bucket,
            "ttl_seconds": self._conf.ttl_seconds,
            "inline_max": self._conf.inline_max,
            "metrics": dict(self._metrics),
            "backend": getattr(self._bus, "get_backend", lambda: "unknown")(),
        }

    # -------- Public API --------

    async def request_with_cache(
        self,
        *,
        subject: str,
        payload: Any,
        version: str = "v1",
        timeout_s: float = 5.0,
        singleflight: bool = True,
        do_request: Optional[Callable[[], Awaitable[bytes]]] = None,
    ) -> Tuple[bytes, bool]:
        """
        Cache wrapper for bus requests.
        If cached: returns (data, True). On miss: performs request, stores, returns (data, False).

        - `payload` can be dict/str/bytes (key is formed from canonical bytes).
        - `do_request` may be provided; if omitted, we'll use self._bus.request(subject, bytes, timeout).
        """
        await self._ensure_ready()
        key = _stable_key(subject, payload, version=version)

        # 1) Fast path
        hit = await self._kv_get(key)
        if hit is not None:
            self._metrics["hits"] += 1
            return hit, True

        self._metrics["misses"] += 1

        # 2) Singleflight lock
        lock_key = f"__lock__/{key}"
        have_lock = False
        if singleflight:
            have_lock = await self._kv_try_create(lock_key, b"1")
            if not have_lock:
                # Wait briefly for another worker to fill
                self._metrics["waits"] += 1
                waited = await self._wait_for_fill(key, timeout_s)
                if waited is not None:
                    self._metrics["hits"] += 1
                    return waited, True

        try:
            # 3) Perform the request
            if do_request is not None:
                data = await do_request()
                if isinstance(data, tuple):
                    # some wrappers may return (Msg, ...) or similar—handle common cases
                    data = data[0]
                if not isinstance(data, (bytes, bytearray, memoryview)):
                    # if someone returned a message object
                    data = getattr(data, "data", data)
                data = bytes(data or b"")
            else:
                # use the injected bus
                if not self._bus or not hasattr(self._bus, "request"):
                    raise RuntimeError("No bus available and no do_request provided")
                req_bytes = _json_bytes(payload)
                msg = await self._bus.request(subject, req_bytes, timeout=timeout_s)
                data = bytes(getattr(msg, "data", b"") or b"")

            # 4) Store and return
            await self._kv_put(key, data)
            self._metrics["stores"] += 1
            return data, False

        finally:
            if have_lock:
                await self._kv_delete(lock_key)

    async def get_or_compute(
        self,
        *,
        key_subject: str,
        key_payload: Any,
        compute: Callable[[], Awaitable[bytes]],
        version: str = "v1",
        timeout_s: float = 5.0,
        singleflight: bool = True,
    ) -> Tuple[bytes, bool]:
        """
        Generic variant for caching arbitrary async computations (not tied to bus).
        """
        async def _do():
            return await compute()

        return await self.request_with_cache(
            subject=key_subject,
            payload=key_payload,
            version=version,
            timeout_s=timeout_s,
            singleflight=singleflight,
            do_request=_do,
        )

    async def get_raw(self, subject: str, payload: Any, *, version: str = "v1") -> Optional[bytes]:
        await self._ensure_ready()
        key = _stable_key(subject, payload, version=version)
        return await self._kv_get(key)

    async def invalidate(self, subject: str, payload: Any, *, version: str = "v1") -> None:
        await self._ensure_ready()
        key = _stable_key(subject, payload, version=version)
        await self._kv_delete(key)

    # -------- Internals: JetStream setup --------

    async def _ensure_ready(self):
        if self._ready:
            return
        global _nats, _js_api, KeyValueConfig, ObjectStoreConfig, StorageType
        if _nats is None:
            import nats as _nats  # type: ignore

        if _js_api is None:
            from nats.js.api import KeyValueConfig, ObjectStoreConfig, StorageType  # type: ignore
            _js_api = True

        # Try to get JetStream from the injected bus
        js = None
        nc = None

        # Common patterns we support:
        # - bus.js  (direct JS context)
        # - bus.get_js()
        # - bus.nc.jetstream()
        candidate_js = getattr(self._bus, "js", None)
        if candidate_js:
            js = candidate_js
            nc = getattr(self._bus, "nc", None)

        if js is None and hasattr(self._bus, "get_js"):
            try:
                js = await self._bus.get_js()
                nc = getattr(self._bus, "nc", None)
            except Exception:
                js = None

        if js is None and hasattr(self._bus, "nc"):
            try:
                nc = self._bus.nc
                if nc:
                    js = nc.jetstream()
            except Exception:
                js = None

        # Fallback: self-connect (optional)
        if js is None and self._conf.connect_if_missing:
            servers = self._conf.nats_servers
            if isinstance(servers, str):
                servers = [servers]
            nc = await _nats.connect(servers=servers)
            js = nc.jetstream()
            self._own_conn = True

        if js is None:
            raise RuntimeError("CacheService: JetStream context not available (no bus.js and no self-connect)")

        self._js = js
        self._nc = nc

        # Ensure KV bucket
        try:
            self._kv = await self._js.key_value(self._conf.bucket)
        except Exception:
            from nats.js.api import KeyValueConfig, StorageType  # type: ignore
            self._kv = await self._js.create_key_value(
                KeyValueConfig(
                    bucket=self._conf.bucket,
                    history=1,
                    ttl=timedelta(seconds=self._conf.ttl_seconds),
                    storage=StorageType.FILE,
                    max_value_size=self._conf.inline_max + 2048,
                )
            )

        # Ensure Object Store for large values
        try:
            self._objs = await self._js.object_store(self._conf.blobs_bucket)
        except Exception:
            from nats.js.api import ObjectStoreConfig, StorageType  # type: ignore
            self._objs = await self._js.create_object_store(
                ObjectStoreConfig(
                    bucket=self._conf.blobs_bucket, storage=StorageType.FILE
                )
            )

        self._ready = True

    # -------- Internals: KV helpers --------

    async def _kv_get(self, key: str) -> Optional[bytes]:
        try:
            e = await self._kv.get(key)
        except Exception:
            return None
        if not e:
            return None

        # New format: JSON meta with kind inline|blob
        try:
            meta = json.loads(e.value.decode("utf-8"))
            kind = meta.get("kind")
            if kind == "inline":
                data_b64 = meta.get("data_b64")
                if data_b64 is not None:
                    return base64.b64decode(data_b64.encode("ascii"))
                # legacy 'data' latin1 path
                data = meta.get("data")
                return data.encode("latin1") if isinstance(data, str) else None
            if kind == "blob":
                name = meta.get("name")
                if not name:
                    return None
                obj = await self._objs.get(name)
                return await obj.read()
        except Exception:
            # Back-compat: raw bytes stored directly
            return e.value
        return None

    async def _kv_put(self, key: str, data: bytes) -> None:
        if len(data) <= self._conf.inline_max:
            meta = {
                "kind": "inline",
                "ts": time.time(),
                "data_b64": base64.b64encode(data).decode("ascii"),
            }
            await self._kv.put(key, json.dumps(meta).encode("utf-8"))
            return

        # Blob path
        name = f"{key}-{int(time.time())}"
        await self._objs.put(name, data)
        meta = {"kind": "blob", "ts": time.time(), "name": name}
        await self._kv.put(key, json.dumps(meta).encode("utf-8"))

    async def _kv_delete(self, key: str) -> None:
        try:
            await self._kv.delete(key)
        except Exception:
            pass

    async def _kv_try_create(self, key: str, value: bytes) -> bool:
        """Return True iff the key was created (i.e., we won the lock)."""
        try:
            await self._kv.create(key, value)
            self._metrics["locks"] += 1
            return True
        except Exception:
            return False

    async def _wait_for_fill(self, key: str, timeout_s: float) -> Optional[bytes]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            hit = await self._kv_get(key)
            if hit is not None:
                return hit
            await asyncio.sleep(0.05)
        return None
