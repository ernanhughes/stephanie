# stephanie/services/bus/idempotency.py
from __future__ import annotations

import asyncio
import os
import time
from typing import Set


class IdempotencyStore:
    """Base interface for idempotency stores."""
    
    async def seen(self, key: str) -> bool:
        """Check if this key has been processed before."""
        raise NotImplementedError
        
    async def mark(self, key: str) -> None:
        """Mark this key as processed."""
        raise NotImplementedError


class InMemoryIdempotencyStore(IdempotencyStore):
    """In-memory idempotency store for development."""
    
    def __init__(self, ttl_sec: int = 3600):
        self._seen: Set[str] = set()
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl_sec
        self._lock = asyncio.Lock()
        
    async def seen(self, key: str) -> bool:
        now = time.time()
        async with self._lock:
            # Clean up stale entries occasionally
            if len(self._timestamps) > 1000:
                stale = [k for k, ts in self._timestamps.items() if now - ts > self._ttl]
                for k in stale:
                    self._seen.discard(k)
                    self._timestamps.pop(k, None)
                    
            return key in self._seen
            
    async def mark(self, key: str) -> None:
        async with self._lock:
            self._seen.add(key)
            self._timestamps[key] = time.time()


class JsonlIdempotencyStore(IdempotencyStore):
    """File-based idempotency store for simple persistence."""
    
    def __init__(self, path: str = "./data/idempotency/keys.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._path = path
        self._seen_set: Set[str] = set()
        self._lock = asyncio.Lock()
        
        # Load existing keys
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    self._seen_set.add(line.strip())
        except FileNotFoundError:
            pass
            
    async def seen(self, key: str) -> bool:
        return key in self._seen_set
        
    async def mark(self, key: str) -> None:
        if key in self._seen_set:
            return
            
        async with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(key + "\n")
        self._seen_set.add(key)


class NatsKVIdempotencyStore(IdempotencyStore):
    """NATS KV-based idempotency store for production."""
    
    def __init__(self, js, bucket: str = "idempotency"):
        self._js = js
        self._bucket = bucket
        self._kv = None
        self._lock = asyncio.Lock()
        
    async def _ensure_kv(self):
        """Ensure KV bucket is available."""
        if self._kv is not None:
            return
            
        try:
            self._kv = await self._js.create_key_value(bucket=self._bucket)
        except Exception:
            self._kv = await self._js.key_value(self._bucket)
            
    async def seen(self, key: str) -> bool:
        """Check if this key has been processed."""
        await self._ensure_kv()
        try:
            entry = await self._kv.get(key)
            return entry is not None
        except Exception:
            return False
            
    async def mark(self, key: str) -> None:
        """Mark this key as processed."""
        await self._ensure_kv()
        async with self._lock:
            try:
                await self._kv.put(key, b"1")
            except Exception as e:
                print(f"Failed to mark idempotency key {key}: {str(e)}")