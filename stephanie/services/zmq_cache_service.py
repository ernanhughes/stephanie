# stephanie/services/cache/zmq_cache_service.py
import asyncio
import sqlite3
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from cachetools import TTLCache

from stephanie.services.service_protocol import Service
from stephanie.utils.json_sanitize import dumps_safe


class ZmqCacheService(Service):
    """
    ZeroMQ-native cache service with two-tier storage:
    - L1: In-memory LRU+TTL
    - L2: SQLite circular buffer

    Features:
    - Singleflight protection
    - Circuit breaker around L2
    - Pluggable as a ZMQ bus request middleware
    """

    def __init__(self, cfg: Dict, memory: Any, logger: Any):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self._ready = False

        # Config with sane defaults
        self.cache_cfg = {
            "l1_size": self.cfg.get("l1_size", 50_000),
            "l2_size": self.cfg.get("l2_size", 2_000_000),
            "ttl_seconds": self.cfg.get("ttl_seconds", 24 * 60 * 60),
            "singleflight_timeout": self.cfg.get("singleflight_timeout", 5.0),
            "enable": self.cfg.get("enable", True),
            "db_path": self.cfg.get("db_path", "data/cache/zmq_cache.db"),
            "circuit_breaker_threshold": self.cfg.get("circuit_breaker_threshold", 10),
            "circuit_breaker_reset": self.cfg.get("circuit_breaker_reset", 30.0),
            "no_cache_subjects": self.cfg.get(
                "no_cache_subjects", ["results.", "debug."]
            ),
        }

        # L1 cache (in-memory)
        self._l1_cache = TTLCache(
            maxsize=self.cache_cfg["l1_size"],
            ttl=self.cache_cfg["ttl_seconds"],
        )

        # Circuit breaker state
        self._circuit_open = False
        self._circuit_failures = 0
        self._circuit_last_failure = 0.0

        # Singleflight state
        self._in_flight: Dict[str, asyncio.Future[bytes]] = {}
        self._lock = asyncio.Lock()

        # Metrics
        self._metrics = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "waits": 0,
            "stores": 0,
            "db_errors": 0,
            "circuit_breaker_trips": 0,
        }

        self._bus = None
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def name(self) -> str:
        return "zmq-cache-v1"

    # -------- Service lifecycle --------

    def set_bus(self, bus: Any) -> None:
        """
        Attach to a ZmqKnowledgeBus (or any bus implementing wrap_request).
        """
        self._bus = bus
        # if hasattr(bus, "wrap_request"):
            # self._bus.wrap_request(self._cache_request_middleware)

    def initialize(self, **kwargs) -> None:
        """Synchronous init; lazy DB setup happens on first use."""
        if kwargs:
            self.cache_cfg.update(kwargs)

    async def _ensure_ready(self) -> None:
        """Lazy initialization of SQLite database."""
        if self._ready:
            return

        try:
            import os

            os.makedirs(
                os.path.dirname(self.cache_cfg["db_path"]), exist_ok=True
            )

            # Autocommit mode: isolation_level=None
            self._conn = sqlite3.connect(
                self.cache_cfg["db_path"],
                check_same_thread=False,
                timeout=5.0,
                isolation_level=None,
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA cache_size=-20000;")  # ~20MB

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL
                );
                """
            )

            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_accessed ON cache(accessed_at);"
            )

            self._conn.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS maintain_capacity 
                AFTER INSERT ON cache 
                WHEN (SELECT COUNT(*) FROM cache) > {self.cache_cfg['l2_size']}
                BEGIN
                    DELETE FROM cache 
                    WHERE rowid IN (
                        SELECT rowid FROM cache 
                        ORDER BY accessed_at ASC 
                        LIMIT 100
                    );
                END;
                """
            )

            self._ready = True
            self.logger.info(
                "[ZmqCache] Ready: L1=%d, L2=%d, TTL=%ds",
                self.cache_cfg["l1_size"],
                self.cache_cfg["l2_size"],
                self.cache_cfg["ttl_seconds"],
            )
        except Exception as e:
            self.logger.error("[ZmqCache] Failed to initialize DB: %s", e)
            raise

    def shutdown(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._ready = False

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._ready else "initializing",
            "l1_size": self.cache_cfg["l1_size"],
            "l2_size": self.cache_cfg["l2_size"],
            "l1_entries": len(self._l1_cache),
            "l2_entries": self._get_l2_count() if self._ready else 0,
            "metrics": dict(self._metrics),
            "circuit_open": self._circuit_open,
            "ready": self._ready,
        }

    # -------- Core API --------

    async def request_with_cache(
        self,
        *,
        subject: str,
        payload: Any,
        version: str = "v1",
        timeout_s: float = 5.0,
        singleflight: bool = True,
        do_request: Optional[Callable[[], Awaitable[Any]]] = None,
    ) -> Tuple[bytes, bool]:
        """
        Primary cache interface.

        Returns: (result_bytes, is_cached)
        """
        if not self.cache_cfg["enable"]:
            # Passthrough
            raw = await self._do_raw_request(subject, payload, timeout_s, do_request)
            return self._normalize_to_bytes(raw), False

        await self._ensure_ready()
        key = self._stable_key(subject, payload, version=version)

        # L1
        if key in self._l1_cache:
            self._metrics["l1_hits"] += 1
            self.logger.debug("[ZmqCache] L1 HIT key=%s", key)
            return self._l1_cache[key], True

        # L2
        l2_value = await self._get_l2(key)
        if l2_value is not None:
            self._l1_cache[key] = l2_value
            self._metrics["l2_hits"] += 1
            self.logger.debug("[ZmqCache] L2 HIT key=%s", key)
            return l2_value, True

        # Miss
        self._metrics["misses"] += 1

        loop = asyncio.get_running_loop()

        # Singleflight: re-use in-flight work
        if singleflight:
            async with self._lock:
                if key in self._in_flight:
                    self._metrics["waits"] += 1
                    fut = self._in_flight[key]
                    result = await asyncio.wait_for(
                        fut, timeout=self.cache_cfg["singleflight_timeout"]
                    )
                    return result, False

            fut: asyncio.Future[bytes] = loop.create_future()
            async with self._lock:
                self._in_flight[key] = fut
        else:
            fut = None

        try:
            start = time.time()
            raw = await self._do_raw_request(subject, payload, timeout_s, do_request)
            result = self._normalize_to_bytes(raw)

            await self._put(key, result)
            self._metrics["stores"] += 1

            elapsed = time.time() - start
            if elapsed > 1.0:
                self.logger.warning(
                    "[ZmqCache] Slow request subject=%s took %.2fs",
                    subject,
                    elapsed,
                )

            # Complete any waiters
            if fut and not fut.done():
                fut.set_result(result)

            return result, False

        except Exception as e:
            # Circuit breaker around L2 operations
            self._circuit_failures += 1
            self._circuit_last_failure = time.time()
            if (
                self._circuit_failures
                >= self.cache_cfg["circuit_breaker_threshold"]
            ):
                self._circuit_open = True
                self._metrics["circuit_breaker_trips"] += 1
                self.logger.error(
                    "[ZmqCache] CIRCUIT OPEN after %d failures",
                    self._circuit_failures,
                )
            raise e
        finally:
            if singleflight:
                async with self._lock:
                    self._in_flight.pop(key, None)

    async def get_raw(
        self, subject: str, payload: Any, *, version: str = "v1"
    ) -> Optional[bytes]:
        if not self.cache_cfg["enable"]:
            return None

        await self._ensure_ready()
        key = self._stable_key(subject, payload, version=version)

        if key in self._l1_cache:
            self._metrics["l1_hits"] += 1
            return self._l1_cache[key]

        return await self._get_l2(key)

    async def invalidate(
        self, subject: str, payload: Any, *, version: str = "v1"
    ) -> None:
        if not self.cache_cfg["enable"] or not self._ready:
            return

        key = self._stable_key(subject, payload, version=version)

        if key in self._l1_cache:
            del self._l1_cache[key]

        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._conn.execute(
                    "DELETE FROM cache WHERE key = ?", (key,)
                ),
            )
        except Exception as e:
            self.logger.error("[ZmqCache] DB delete failed: %s", e)
            self._metrics["db_errors"] += 1

    # -------- ZMQ middleware integration --------

    async def _cache_request_middleware(
        self,
        next_handler: Callable[..., Awaitable[Any]],
        subject: str,
        payload: Any,
        **kwargs,
    ):
        """
        Bus.wrap_request middleware. MUST return the same type that
        callers expect from bus.request (usually dict).
        """
        # 1) Skip for no-cache subjects
        for prefix in self.cache_cfg["no_cache_subjects"]:
            if subject.startswith(prefix):
                return await next_handler(subject, payload, **kwargs)

        # 2) Only cache "request" method, not publish
        if kwargs.get("method") != "request":
            return await next_handler(subject, payload, **kwargs)

        timeout = kwargs.get("timeout", 5.0)

        # 3) Use cache, then convert back to dict/structured form
        result_bytes, _ = await self.request_with_cache(
            subject=subject,
            payload=payload,
            timeout_s=timeout,
            do_request=lambda: next_handler(subject, payload, **kwargs),
        )

        # Try to decode JSON back into dict; fall back to bytes
        try:
            text = result_bytes.decode("utf-8")
            import json

            return json.loads(text)
        except Exception:
            return result_bytes

    # -------- Internal helpers --------

    async def _do_raw_request(
        self,
        subject: str,
        payload: Any,
        timeout_s: float,
        do_request: Optional[Callable[[], Awaitable[Any]]] = None,
    ) -> Any:
        if do_request is not None:
            return await do_request()
        if self._bus and hasattr(self._bus, "request"):
            return await self._bus.request(subject, payload, timeout=timeout_s)
        raise RuntimeError("No request method available for ZmqCacheService")

    def _normalize_to_bytes(self, data: Any) -> bytes:
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, str):
            return data.encode("utf-8")
        if isinstance(data, dict):
            return dumps_safe(data).encode("utf-8")
        return str(data).encode("utf-8")

    def _stable_key(self, subject: str, payload: Any, *, version: str = "v1") -> str:
        import hashlib

        h = hashlib.blake2s(digest_size=16)
        h.update(subject.encode("utf-8"))
        h.update(b"\x00")
        h.update(self._json_bytes(payload))
        h.update(b"\x00")
        h.update(version.encode("utf-8"))
        return h.hexdigest()

    def _json_bytes(self, obj: Any) -> bytes:
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return bytes(obj)
        if isinstance(obj, str):
            return obj.encode("utf-8")
        return dumps_safe(obj).encode("utf-8")

    async def _get_l2(self, key: str) -> Optional[bytes]:
        if not self._ready:
            return None

        # Circuit breaker: half-open logic
        if self._circuit_open:
            if (
                time.time() - self._circuit_last_failure
                < self.cache_cfg["circuit_breaker_reset"]
            ):
                return None
            # Reset
            self._circuit_open = False
            self._circuit_failures = 0

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._conn.execute(
                    "SELECT value FROM cache WHERE key = ? AND accessed_at > ?",
                    (key, time.time() - self.cache_cfg["ttl_seconds"]),
                ).fetchone(),
            )

            if result:
                # Touch accessed_at
                await loop.run_in_executor(
                    None,
                    lambda: self._conn.execute(
                        "UPDATE cache SET accessed_at = ? WHERE key = ?",
                        (time.time(), key),
                    ),
                )
                return result[0]

            return None
        except Exception as e:
            self.logger.error("[ZmqCache] L2 lookup failed: %s", e)
            self._metrics["db_errors"] += 1
            self._circuit_failures += 1
            self._circuit_last_failure = time.time()
            if (
                self._circuit_failures
                >= self.cache_cfg["circuit_breaker_threshold"]
            ):
                self._circuit_open = True
                self._metrics["circuit_breaker_trips"] += 1
            return None

    async def _put(self, key: str, value: bytes) -> None:
        self._l1_cache[key] = value

        if self._ready and not self._circuit_open:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._conn.execute(
                        """
                        INSERT OR REPLACE INTO cache
                        (key, value, created_at, accessed_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (key, value, time.time(), time.time()),
                    ),
                )
            except Exception as e:
                self.logger.error("[ZmqCache] L2 store failed: %s", e)
                self._metrics["db_errors"] += 1

    def _get_l2_count(self) -> int:
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
