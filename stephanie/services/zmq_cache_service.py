# stephanie/services/cache/zmq_cache_service.py
import asyncio
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Callable, Awaitable
from cachetools import TTLCache, LRUCache
import numpy as np

from stephanie.services.service_protocol import Service
from stephanie.utils.json_sanitize import dumps_safe

class ZmqCacheService(Service):
    """
    ZeroMQ-native cache service with two-tier storage:
    - L1: In-memory LRU cache (fast, volatile)
    - L2: SQLite circular buffer (persistent, larger)
    
    Designed for production with:
    - Singleflight protection
    - Metrics collection
    - Circuit breaking
    - Configurable TTLs
    """
    def __init__(self, cfg: Dict, memory: Any, logger: Any):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self._ready = False
        
        # Config with sane defaults
        self.cache_cfg = {
            "l1_size": self.cfg.get("l1_size", 50_000),       # In-memory entries
            "l2_size": self.cfg.get("l2_size", 2_000_000),    # DB entries (circular)
            "ttl_seconds": self.cfg.get("ttl_seconds", 24*60*60),  # 24 hours
            "singleflight_timeout": self.cfg.get("singleflight_timeout", 5.0),
            "enable": self.cfg.get("enable", True),
            "db_path": self.cfg.get("db_path", "data/cache/zmq_cache.db"),
            "circuit_breaker_threshold": self.cfg.get("circuit_breaker_threshold", 10),
            "circuit_breaker_reset": self.cfg.get("circuit_breaker_reset", 30.0),
        }
        
        # In-memory L1 cache (LRU with TTL)
        self._l1_cache = TTLCache(
            maxsize=self.cache_cfg["l1_size"],
            ttl=self.cache_cfg["ttl_seconds"]
        )
        
        # Circuit breaker state
        self._circuit_open = False
        self._circuit_failures = 0
        self._circuit_last_failure = 0.0
        
        # Singleflight locks
        self._in_flight = {}
        self._lock = asyncio.Lock()
        
        # Metrics
        self._metrics = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "stores": 0,
            "db_errors": 0,
            "circuit_breaker_trips": 0,
        }
        
        # DB connection will be initialized in _ensure_ready()

    @property
    def name(self) -> str:
        return "zmq-cache-v1"

    # -------- Service lifecycle --------
    
    def set_bus(self, bus: Any) -> None:
        self._bus = bus
        # Install ourselves as a middleware
        if hasattr(bus, "wrap_request"):
            bus.wrap_request(self._cache_request_middleware)

    def initialize(self, **kwargs) -> None:
        """Synchronous init; lazy DB setup happens on first use"""
        if kwargs:
            self.cache_cfg.update(kwargs)
        # We'll connect to DB on first use

    async def _ensure_ready(self) -> None:
        """Lazy initialization of SQLite database"""
        if self._ready:
            return
            
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self.cache_cfg["db_path"]), exist_ok=True)
            
            # Connect to SQLite
            self._conn = sqlite3.connect(
                self.cache_cfg["db_path"],
                check_same_thread=False,
                timeout=5.0
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA cache_size=-20000;")  # 20MB cache
            
            # Create tables
            self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL
            );
            """)
            
            self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accessed ON cache(accessed_at);
            """)
            
            # Set up circular buffer trigger
            self._conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS maintain_capacity 
            AFTER INSERT ON cache 
            WHEN (SELECT COUNT(*) FROM cache) > {self.cache_cfg['l2_size']}
            BEGIN
                DELETE FROM cache 
                WHERE rowid IN (
                    SELECT rowid FROM cache ORDER BY accessed_at ASC LIMIT 100
                );
            END;
            """)
            
            self._ready = True
            self.logger.info("[ZmqCache] Ready: L1=%d, L2=%d, TTL=%ds",
                            self.cache_cfg["l1_size"],
                            self.cache_cfg["l2_size"],
                            self.cache_cfg["ttl_seconds"])
                            
        except Exception as e:
            self.logger.error("[ZmqCache] Failed to initialize DB: %s", e)
            raise

    def shutdown(self) -> None:
        """Clean shutdown"""
        if hasattr(self, "_conn") and self._conn:
            try:
                self._conn.close()
            except:
                pass
        self._ready = False

    def health_check(self) -> Dict[str, Any]:
        """Return service health status"""
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
        do_request: Optional[Callable[[], Awaitable[bytes]]] = None,
    ) -> Tuple[bytes, bool]:
        """
        Primary cache interface - works with or without bus integration
        
        Returns (data, is_cached) where:
        - is_cached=True means value came from cache
        - is_cached=False means value came from computation
        """
        if not self.cache_cfg["enable"]:
            # Bypass cache completely
            if do_request:
                return await do_request(), False
            elif self._bus and hasattr(self._bus, "request"):
                return await self._bus.request(subject, payload, timeout=timeout_s), False
            raise RuntimeError("No request method available")

        key = self._stable_key(subject, payload, version=version)
        
        # 1. Check L1 cache (fastest)
        if key in self._l1_cache:
            self._metrics["l1_hits"] += 1
            self.logger.debug("[ZmqCache] L1 HIT key=%s", key)
            return self._l1_cache[key], True

        # 2. Check L2 cache (SQLite)
        l2_value = await self._get_l2(key)
        if l2_value is not None:
            # Promote to L1
            self._l1_cache[key] = l2_value
            self._metrics["l2_hits"] += 1
            self.logger.debug("[ZmqCache] L2 HIT key=%s", key)
            return l2_value, True

        # 3. Cache miss - handle singleflight
        self._metrics["misses"] += 1
        if singleflight:
            async with self._lock:
                if key in self._in_flight:
                    # Another request is already in flight
                    self._metrics["waits"] += 1
                    return await self._in_flight[key], False
            
            # Claim this key for processing
            fut = asyncio.get_event_loop().create_future()
            async with self._lock:
                self._in_flight[key] = fut

        try:
            # 4. Perform the actual request
            start = time.time()
            try:
                if do_request:
                    data = await do_request()
                elif self._bus and hasattr(self._bus, "request"):
                    data = await self._bus.request(subject, payload, timeout=timeout_s)
                else:
                    raise RuntimeError("No request method available")
                
                # Normalize to bytes
                if isinstance(data, (bytes, bytearray)):
                    result = bytes(data)
                elif isinstance(data, str):
                    result = data.encode("utf-8")
                elif isinstance(data, dict):
                    result = dumps_safe(data).encode("utf-8")
                else:
                    result = str(data).encode("utf-8")
                    
                # 5. Store in BOTH caches
                await self._put(key, result)
                self._metrics["stores"] += 1
                
                # 6. Record success time
                elapsed = time.time() - start
                if elapsed > 1.0:  # Log slow requests
                    self.logger.warning(
                        "[ZmqCache] Slow request subject=%s took %.2fs",
                        subject, elapsed
                    )
                
                return result, False
                
            except Exception as e:
                # Circuit breaker logic
                self._circuit_failures += 1
                self._circuit_last_failure = time.time()
                
                if self._circuit_failures >= self.cache_cfg["circuit_breaker_threshold"]:
                    self._circuit_open = True
                    self._metrics["circuit_breaker_trips"] += 1
                    self.logger.error(
                        "[ZmqCache] CIRCUIT OPEN after %d failures",
                        self._circuit_failures
                    )
                
                raise
                
        finally:
            # Clean up singleflight
            if singleflight:
                async with self._lock:
                    if key in self._in_flight:
                        del self._in_flight[key]

    async def get_raw(self, subject: str, payload: Any, *, version: str = "v1") -> Optional[bytes]:
        """Check cache without triggering computation"""
        if not self.cache_cfg["enable"] or not self._ready:
            return None
            
        key = self._stable_key(subject, payload, version=version)
        
        # Check L1
        if key in self._l1_cache:
            self._metrics["l1_hits"] += 1
            return self._l1_cache[key]
            
        # Check L2
        return await self._get_l2(key)

    async def invalidate(self, subject: str, payload: Any, *, version: str = "v1") -> None:
        """Remove entry from ALL cache layers"""
        if not self.cache_cfg["enable"] or not self._ready:
            return
            
        key = self._stable_key(subject, payload, version=version)
        
        # Remove from L1
        if key in self._l1_cache:
            del self._l1_cache[key]
            
        # Remove from L2
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            )
        except Exception as e:
            self.logger.error("[ZmqCache] DB delete failed: %s", e)
            self._metrics["db_errors"] += 1

    # -------- ZeroMQ Middleware Integration --------
    
    async def _cache_request_middleware(self, next_handler, subject, payload, **kwargs):
        """
        ZeroMQ bus middleware that automatically caches requests
        Only caches requests to specific subjects (configurable)
        """
        # Skip caching for certain subjects
        no_cache_subjects = self.cfg.get("no_cache_subjects", ["results.", "debug."])
        if any(subject.startswith(prefix) for prefix in no_cache_subjects):
            return await next_handler(subject, payload, **kwargs)
            
        # Only cache GET-style requests (not publishes)
        if kwargs.get("method") != "request":
            return await next_handler(subject, payload, **kwargs)
            
        # Use cache for this request
        timeout = kwargs.get("timeout", 5.0)
        return await self.request_with_cache(
            subject=subject,
            payload=payload,
            timeout_s=timeout,
            do_request=lambda: next_handler(subject, payload, **kwargs)
        )

    # -------- Cache Internals --------
    
    def _stable_key(self, subject: str, payload: Any, *, version: str = "v1") -> str:
        """Create stable cache key from subject + payload"""
        import hashlib
        h = hashlib.blake2s(digest_size=16)
        h.update(subject.encode("utf-8"))
        h.update(b"\x00")
        h.update(self._json_bytes(payload))
        h.update(b"\x00")
        h.update(version.encode("utf-8"))
        return h.hexdigest()

    def _json_bytes(self, obj: Any) -> bytes:
        """Convert any object to canonical JSON bytes"""
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return bytes(obj)
        if isinstance(obj, str):
            return obj.encode("utf-8")
        return dumps_safe(obj).encode("utf-8")

    async def _get_l2(self, key: str) -> Optional[bytes]:
        """Get value from L2 (SQLite) cache"""
        if not self._ready or self._circuit_open:
            return None
            
        try:
            # Circuit breaker check
            if self._circuit_open:
                if time.time() - self._circuit_last_failure < self.cache_cfg["circuit_breaker_reset"]:
                    return None
                # Reset circuit breaker
                self._circuit_open = False
                self._circuit_failures = 0
                
            # Query DB
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._conn.execute(
                    "SELECT value FROM cache WHERE key = ? AND accessed_at > ?",
                    (key, time.time() - self.cache_cfg["ttl_seconds"])
                ).fetchone()
            )
            
            if result:
                # Update access time
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._conn.execute(
                        "UPDATE cache SET accessed_at = ? WHERE key = ?",
                        (time.time(), key)
                    )
                )
                return result[0]
                
            return None
            
        except Exception as e:
            self.logger.error("[ZmqCache] L2 lookup failed: %s", e)
            self._metrics["db_errors"] += 1
            
            # Circuit breaker logic
            self._circuit_failures += 1
            self._circuit_last_failure = time.time()
            
            if self._circuit_failures >= self.cache_cfg["circuit_breaker_threshold"]:
                self._circuit_open = True
                self._metrics["circuit_breaker_trips"] += 1
                
            return None

    async def _put(self, key: str, value: bytes) -> None:
        """Store value in BOTH cache layers"""
        # Update L1
        self._l1_cache[key] = value
        
        # Update L2 (in background)
        if self._ready and not self._circuit_open:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._conn.execute(
                        "INSERT OR REPLACE INTO cache (key, value, created_at, accessed_at) "
                        "VALUES (?, ?, ?, ?)",
                        (key, value, time.time(), time.time())
                    )
                )
            except Exception as e:
                self.logger.error("[ZmqCache] L2 store failed: %s", e)
                self._metrics["db_errors"] += 1

    def _get_l2_count(self) -> int:
        """Get current number of entries in L2 cache"""
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return row[0] if row else 0
        except:
            return 0