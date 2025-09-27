# stephanie/services/bus/idempotency.py
"""
Idempotency Store Implementations

Provides multiple implementations for tracking processed messages to ensure
idempotent processing across the event bus system.

Idempotency is critical for:
- Preventing duplicate processing of the same message
- Ensuring exactly-once semantics where possible
- Maintaining system consistency
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import Set


class IdempotencyStore(ABC):
    """Abstract base class for idempotency stores."""
    
    @abstractmethod
    async def seen(self, key: str) -> bool:
        """
        Check if this key has been processed before.
        
        Args:
            key: Unique identifier for the message/event
            
        Returns:
            bool: True if key has been processed, False otherwise
        """
        raise NotImplementedError
        
    @abstractmethod
    async def mark(self, key: str) -> None:
        """
        Mark this key as processed.
        
        Args:
            key: Unique identifier for the message/event
        """
        raise NotImplementedError


class InMemoryIdempotencyStore(IdempotencyStore):
    """
    In-memory idempotency store for development and testing.
    
    Features:
    - Fast in-process storage
    - Automatic TTL-based cleanup
    - Thread-safe operations
    
    Not suitable for production as it doesn't persist across restarts
    or work in distributed environments.
    """
    
    def __init__(self, ttl_sec: int = 3600):
        """
        Initialize the in-memory store.
        
        Args:
            ttl_sec: Time-to-live for keys in seconds (default: 1 hour)
        """
        self._seen: Set[str] = set()
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl_sec
        self._lock = asyncio.Lock()
        
    async def seen(self, key: str) -> bool:
        """
        Check if key exists and hasn't expired.
        
        Performs occasional cleanup of expired entries when the store
        grows beyond 1000 entries to prevent memory leaks.
        """
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
        """Mark a key as processed with current timestamp."""
        async with self._lock:
            self._seen.add(key)
            self._timestamps[key] = time.time()


class JsonlIdempotencyStore(IdempotencyStore):
    """
    File-based idempotency store for simple persistence.
    
    Features:
    - Persistent storage across restarts
    - Simple JSONL file format
    - Suitable for single-node deployments
    
    Not suitable for distributed systems due to file locking limitations.
    """
    
    def __init__(self, path: str = "./data/idempotency/keys.jsonl"):
        """
        Initialize the JSONL file store.
        
        Args:
            path: File path for storing processed keys
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._path = path
        self._seen_set: Set[str] = set()
        self._lock = asyncio.Lock()
        
        # Load existing keys
        try:
            if not os.path.exists(self._path):
                with open(self._path, "w", encoding="utf-8") as f:
                    pass  # Create empty file if it doesn't exist
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    self._seen_set.add(line.strip())
        except FileNotFoundError:
            pass
            
    async def seen(self, key: str) -> bool:
        """Check if key exists in the stored set."""
        return key in self._seen_set
        
    async def mark(self, key: str) -> None:
        """Add key to the stored set and persist to file."""
        if key in self._seen_set:
            return
            
        async with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(key + "\n")
            self._seen_set.add(key)


class NatsKVIdempotencyStore(IdempotencyStore):
    """
    NATS KV-based idempotency store for production.
    
    Features:
    - Distributed storage suitable for clusters
    - Persistent across restarts
    - High performance
    - Automatic expiration (via NATS KV TTL)
    
    Requires NATS JetStream with Key-Value functionality.
    """
    
    def __init__(self, js, bucket: str = "idempotency"):
        """
        Initialize the NATS KV store.
        
        Args:
            js: NATS JetStream context
            bucket: KV bucket name for idempotency keys
        """
        self._js = js
        self._bucket = bucket
        self._kv = None
        self._lock = asyncio.Lock()
        
    async def _ensure_kv(self):
        """Ensure KV bucket is available, create if needed."""
        if self._kv is not None:
            return
            
        try:
            self._kv = await self._js.key_value(self._bucket)
        except Exception:
            # Bucket doesn't exist, create it
            self._kv = await self._js.create_key_value(bucket=self._bucket)
            
    async def seen(self, key: str) -> bool:
        """Check if this key has been processed using NATS KV."""
        await self._ensure_kv()
        try:
            entry = await self._kv.get(key)
            return entry is not None
        except Exception:
            return False
            
    async def mark(self, key: str) -> None:
        """Mark this key as processed using NATS KV."""
        await self._ensure_kv()
        async with self._lock:
            try:
                await self._kv.put(key, b"1")
            except Exception as e:
                # Log error but don't fail the operation
                # In a production system, you might want to retry or use a fallback
                print(f"Failed to mark idempotency key {key}: {str(e)}")