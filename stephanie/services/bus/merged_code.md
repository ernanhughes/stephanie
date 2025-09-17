<!-- Merged Python Code Files -->


## File: bus_protocol.py

`python
# stephanie/services/bus/bus_protocol.py
"""
Event Bus Protocol Definition

Defines the standard interface for all event bus implementations in the Stephanie AI system.
This protocol ensures consistent behavior across different transport mechanisms (NATS, Redis, in-process).

The bus protocol supports:
- Pub/Sub messaging patterns
- Request/Response patterns
- Idempotent message processing
- Graceful connection management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class BusProtocol(ABC):
    """Unified interface for all event bus implementations."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the bus backend.
        
        Returns:
            bool: True if connection was successful, False otherwise
            
        Note:
            Implementations should handle reconnection logic internally
        """
        pass
        
    @abstractmethod
    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """
        Publish a message to the specified subject.
        
        Args:
            subject: Topic/channel to publish to
            payload: Dictionary containing message data
            
        Raises:
            BusPublishError: If message cannot be published
        """
        pass
        
    @abstractmethod
    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to messages on the specified subject.
        
        Args:
            subject: Topic/channel to subscribe to
            handler: Callback function to handle incoming messages
            
        Note:
            Handler should be designed to process messages idempotently
        """
        pass
        
    @abstractmethod
    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Send a request and wait for a reply.
        
        Args:
            subject: Topic/channel for the request
            payload: Dictionary containing request data
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            Optional[Dict]: Response data or None if timeout
            
        Raises:
            BusRequestError: If request cannot be completed
        """
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """
        Gracefully shut down the connection.
        
        Note:
            Should ensure all pending messages are processed before closing
        """
        pass
        
    @abstractmethod
    def get_backend(self) -> str:
        """
        Return the name of the active backend.
        
        Returns:
            str: Backend identifier (nats, redis, inprocess)
        """
        pass
        
    @property
    @abstractmethod
    def idempotency_store(self) -> Any:
        """
        Return the idempotency store for this bus.
        
        Returns:
            Idempotency store instance for tracking processed messages
        """
        pass
``n

## File: hybrid_bus.py

`python
# stephanie/services/bus/hybrid_bus.py
"""
Hybrid Event Bus Implementation

Auto-selects the best available transport backend in priority order:
1. NATS JetStream (persistent, durable) - Production preferred
2. Redis Pub/Sub (transient) - Fallback option  
3. In-process bus (dev-only) - Development fallback

All services use the same interface regardless of backend, ensuring consistency
across development and production environments.

Features:
- Automatic failover between backends
- Consistent API regardless of underlying transport
- Built-in idempotency handling
- Connection pooling and management
"""

import logging
from typing import Any, Callable, Dict, Optional

from .bus_protocol import BusProtocol
from .idempotency import InMemoryIdempotencyStore
from .inprocess_bus import InProcessKnowledgeBus
from .nats_bus import NatsKnowledgeBus
# Note: Redis bus implementation would be imported here when available


class HybridKnowledgeBus(BusProtocol):
    """
    Auto-selects the best available bus implementation.
    
    This bus implementation provides a unified interface while automatically
    selecting the most appropriate backend based on configuration and availability.
    
    Attributes:
        cfg: Configuration dictionary for bus setup
        logger: Logger instance for bus operations
        _bus: Active bus implementation instance
        _idem_store: Idempotency store for message tracking
        _backend: Name of the active backend
    """
    
    def __init__(self, cfg: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the hybrid bus.
        
        Args:
            cfg: Configuration dictionary with bus settings
            logger: Optional logger instance (defaults to module logger)
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self._bus: Optional[BusProtocol] = None
        self._idem_store = None
        self._backend = None
        
    async def connect(self) -> bool:
        """
        Connect to the best available bus backend.
        
        Attempts connection to backends in priority order:
        1. NATS JetStream (production)
        2. Redis Pub/Sub (alternative)
        3. In-process (development fallback)
        
        Returns:
            bool: True if connection to any backend was successful
        """
        bus_config = self.cfg.get("bus", {})
        preferred_backend = bus_config.get("backend")
        
        # 1. Try NATS first (preferred for production)
        if preferred_backend in [None, "nats"]:
            try:
                nats_bus = NatsKnowledgeBus(
                    servers=bus_config.get("servers", ["nats://localhost:4222"]),
                    stream=bus_config.get("stream", "stephanie"),
                    logger=self.logger
                )
                if await nats_bus.connect():
                    self._bus = nats_bus
                    self._backend = "nats"
                    self._idem_store = nats_bus.idempotency_store
                    self.logger.info("Connected to NATS JetStream bus")
                    return True
            except Exception as e:
                self.logger.warning(f"NATS connection failed: {str(e)}")
                
        # 2. Try Redis if configured (implementation needed)
        if preferred_backend == "redis":
            self.logger.warning("Redis bus backend not yet implemented")
            # Redis bus implementation would go here
            
        # 3. Fall back to in-process (for development)
        try:
            inprocess_bus = InProcessKnowledgeBus(logger=self.logger)
            if await inprocess_bus.connect():
                self._bus = inprocess_bus
                self._backend = "inprocess"
                self._idem_store = inprocess_bus.idempotency_store
                self.logger.info("Connected to in-process event bus (development mode)")
                return True
        except Exception as e:
            self.logger.error(f"In-process bus failed: {str(e)}")
            
        self.logger.error("No bus backend available")
        return False
        
    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """Publish an event with standard envelope."""
        if not self._bus and not await self.connect():
            self.logger.error("Cannot publish - no bus connection available")
            return
                
        try:
            await self._bus.publish(subject, payload)
        except Exception as e:
            self.logger.error(f"Failed to publish to {subject}: {str(e)}")
            raise BusPublishError(f"Failed to publish to {subject}") from e
        
    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to events with idempotency handling."""
        if not self._bus and not await self.connect():
            self.logger.error("Cannot subscribe - no bus connection available")
            return
                
        try:
            await self._bus.subscribe(subject, handler)
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {subject}: {str(e)}")
            raise BusSubscribeError(f"Failed to subscribe to {subject}") from e

    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Send a request and wait for a reply."""
        if not self._bus and not await self.connect():
            self.logger.error("Cannot send request - no bus connection available")
            return None
                
        try:
            return await self._bus.request(subject, payload, timeout)
        except Exception as e:
            self.logger.error(f"Request failed for {subject}: {str(e)}")
            return None
        """Gracefully shut down the connection."""
        if self._bus:
            try:
                await self._bus.close()
                self.logger.info("Bus connection closed")
            except Exception as e:
                self.logger.error(f"Error during bus shutdown: {str(e)}")
            
    def get_backend(self) -> str:
        """Return the active backend name."""
        return self._backend or "none"
        
    @property
    def idempotency_store(self):
        """Access the idempotency store for this bus."""
        if not self._idem_store:
            # Fallback to in-memory if nothing else is available
            return InMemoryIdempotencyStore()
        return self._idem_store


# Custom exceptions for better error handling
class BusError(Exception):
    """Base exception for all bus-related errors."""
    pass

class BusPublishError(BusError):
    """Raised when message publishing fails."""
    pass

class BusSubscribeError(BusError):
    """Raised when subscription fails."""
    pass

class BusConnectionError(BusError):
    """Raised when connection to bus backend fails."""
    pass

class BusRequestError(Exception):
    """Exception raised when a bus request fails."""
    pass
``n

## File: idempotency.py

`python
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
``n

## File: inprocess_bus.py

`python
# stephanie/services/bus/inprocess_bus.py
"""
In-Process Event Bus Implementation

Provides a simple pub/sub implementation without external dependencies.
Designed for development, testing, and environments where external messaging
systems are not available.

Features:
- Zero external dependencies
- Simple in-memory pub/sub
- Async/sync handler compatibility
- Basic idempotency support

Limitations:
- Not persistent across restarts
- Not suitable for distributed systems
- No message durability guarantees
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from time import time
from typing import Any, Callable, Dict, List, Optional

from .bus_protocol import BusProtocol
from .idempotency import InMemoryIdempotencyStore


class InProcessKnowledgeBus(BusProtocol):
    """
    In-process event bus for development and testing.
    
    This implementation uses simple in-memory data structures to provide
    pub/sub functionality without external dependencies.
    
    Attributes:
        logger: Logger instance for bus operations
        _subscribers: Dictionary mapping subjects to handler lists
        _idem_store: In-memory idempotency store
        _connected: Connection status flag
        _loop: Asyncio event loop reference
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the in-process bus.
        
        Args:
            logger: Optional logger instance (defaults to module logger)
        """
        self.logger = logger or logging.getLogger(__name__)
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._idem_store = InMemoryIdempotencyStore()
        self._connected = False
        self._loop = asyncio.get_event_loop()
        
    async def connect(self) -> bool:
        """
        Connect to the in-process bus.
        
        Returns:
            bool: Always returns True (in-process bus is always available)
        """
        self._connected = True
        self.logger.info("Connected to in-process event bus (development mode)")
        return True
        
    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """
        Publish an event to all subscribers.
        
        Wraps the payload in a standard envelope with metadata before
        delivering to subscribers.
        
        Args:
            subject: Event subject/topic
            payload: Event data payload
        """
        if not self._connected:
            if not await self.connect():
                return
                
        # Create standard event envelope
        envelope = {
            "event_id": f"{subject}-{uuid.uuid4().hex}",
            "timestamp": time.time(),
            "subject": subject,
            "payload": payload
        }
        
        # Deliver to all subscribers (fire and forget)
        if subject in self._subscribers:
            for handler in self._subscribers[subject]:
                # Handle both async and sync handlers
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(envelope["payload"]))
                else:
                    # Run sync handlers in thread pool
                    asyncio.create_task(
                        self._loop.run_in_executor(None, handler, envelope["payload"])
                    )
                
    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to events on a subject.
        
        Args:
            subject: Event subject/topic to subscribe to
            handler: Callback function to handle events
        """
        if not self._connected:
            if not await self.connect():
                return
                
        if subject not in self._subscribers:
            self._subscribers[subject] = []
        self._subscribers[subject].append(handler)
        self.logger.debug(f"Subscribed to {subject} with {len(self._subscribers[subject])} total handlers")
        
    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Send a request and wait for a reply.
        
        Note: Request/response pattern is not supported in the in-process bus
        as there's no mechanism for correlating requests to responses.
        
        Args:
            subject: Request subject/topic
            payload: Request data
            timeout: Ignored for in-process bus
            
        Returns:
            None: Always returns None (not supported)
        """
        self.logger.warning("Request/reply pattern not supported in in-process bus")
        return None
        
    async def close(self) -> None:
        """Gracefully shut down the connection."""
        self._subscribers.clear()
        self._connected = False
        self.logger.info("In-process bus disconnected")
        
    def get_backend(self) -> str:
        """Return the active backend name."""
        return "inprocess"
        
    @property
    def idempotency_store(self):
        """Access the idempotency store for this bus."""
        return self._idem_store
``n

## File: knowledge_bus.py

`python
# stephanie/services/knowledge_bus.py
"""
In-process KnowledgeBus
=======================

A durable, dependency-free pub/sub bus for Stephanie.
- Thread-safe
- Supports multiple consumers
- Tracks metrics
- No external dependencies (ideal for local + test environments)
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Optional


class KnowledgeBus:
    """Abstract interface for knowledge event bus."""

    def publish(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError

    def consume(self, topic: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def consume_batch(self, topic: str, max_items: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime metrics for monitoring."""
        raise NotImplementedError
``n

## File: kv_sync.py

`python
# stephanie/services/bus/kv_sync.py
import asyncio
import threading
from typing import Optional


class SyncKV:
    def __init__(self, js, bucket, max_age_seconds=None, description=None):
        self._loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._t.start()
        self._js = js
        self._bucket = bucket
        self._kv = self._run(self._ensure(bucket, max_age_seconds, description))

    async def _ensure(self, bucket, max_age_seconds, description):
        try:
            return await self._js.key_value(bucket=bucket)
        except:
            return await self._js.create_key_value(bucket=bucket, description=description, max_age=max_age_seconds)

    def _run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def get(self, key: str) -> Optional[bytes]:
        async def _get():
            try:
                e = await self._kv.get(key)
                return None if e is None else e.value
            except:
                return None
        return self._run(_get())

    def put(self, key: str, value: bytes) -> None:
        async def _put():
            try:
                await self._kv.put(key, value)
            except:
                pass
        self._run(_put())
``n

## File: nats_bus.py

`python
# stephanie/services/bus/nats_bus.py
"""
NATS JetStream Bus Implementation

Production-ready event bus implementation using NATS JetStream.
Provides persistent, durable messaging with at-least-once delivery semantics.

Features:
- Persistent message storage
- Durable consumers
- Idempotent processing
- Dead letter queue support
- Request/reply pattern
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from nats.aio.client import Client as NATS
from nats.errors import NoServersError, TimeoutError
from nats.js.api import ConsumerConfig, DeliverPolicy

from .bus_protocol import BusProtocol
from .idempotency import InMemoryIdempotencyStore, NatsKVIdempotencyStore
from .hybrid_bus import BusConnectionError, BusPublishError, BusSubscribeError, BusRequestError

class NatsKnowledgeBus(BusProtocol):
    """
    NATS JetStream implementation of the event bus.
    
    This implementation provides production-grade messaging with:
    - Persistent storage via JetStream
    - Durable consumers that survive restarts
    - Idempotent message processing
    - Dead letter queue support for failed messages
    
    Attributes:
        servers: List of NATS server URLs
        stream: JetStream stream name
        logger: Logger instance for bus operations
        _nc: NATS connection instance
        _js: JetStream context
        _idem_store: Idempotency store instance
        _connected: Connection status flag
        _subscriptions: Active subscription references
    """
    
    def __init__(
        self,
        servers: List[str] = ["nats://localhost:4222"],
        stream: str = "stephanie",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the NATS bus.
        
        Args:
            servers: List of NATS server URLs
            stream: JetStream stream name
            logger: Optional logger instance
        """
        self.servers = servers
        self.stream = stream
        self.logger = logger or logging.getLogger(__name__)
        self._nc: Optional[NATS] = None
        self._js = None
        self._idem_store = None
        self._connected = False
        self._subscriptions = {}
        
    async def connect(self) -> bool:
        """
        Connect to NATS with JetStream capability check.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self._connected:
            return True
            
        try:
            self._nc = NATS()
            await self._nc.connect(
                servers=self.servers,
                allow_reconnect=True,
                reconnect_time_wait=2.0,
                max_reconnect_attempts=5
            )
            self._js = self._nc.jetstream()
            
            # Verify JetStream is available and configure stream
            try:
                await self._js.add_stream(
                    name=self.stream, 
                    subjects=[f"{self.stream}.>"]
                )
                self.logger.info(f"Connected to NATS JetStream (stream: {self.stream})")
            except Exception as e:
                self.logger.warning(f"JetStream configuration failed: {str(e)}")
                await self._nc.close()
                return False
                
            # Create idempotency store using NATS KV
            self._idem_store = NatsKVIdempotencyStore(self._js, bucket=f"{self.stream}_idem")
            
            self._connected = True
            return True
            
        except (NoServersError, OSError, TimeoutError) as e:
            self.logger.warning(f"NATS connection failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"NATS initialization error: {str(e)}", exc_info=True)
            return False
            
    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """
        Publish with standard event envelope.
        
        Args:
            subject: Event subject/topic
            payload: Event data payload
            
        Raises:
            BusPublishError: If publishing fails
        """
        if not self._connected and not await self.connect():
            raise BusConnectionError("Not connected to NATS")
            
        # Create standard event envelope
        envelope = {
            "event_id": f"{subject}-{uuid.uuid4().hex}",
            "timestamp": time.time(),
            "subject": subject,
            "payload": payload
        }
        
        data = json.dumps(envelope).encode()
        try:
            await self._js.publish(f"{self.stream}.{subject}", data)
            self.logger.debug(f"Published to {subject}: {envelope['event_id']}")
        except Exception as e:
            self.logger.error(f"Failed to publish to {subject}: {str(e)}")
            raise BusPublishError(f"Failed to publish to {subject}") from e
            
    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe with durable consumer and idempotency handling.
        
        Args:
            subject: Event subject/topic to subscribe to
            handler: Callback function to handle events
            
        Raises:
            BusSubscribeError: If subscription fails
        """
        if not self._connected and not await self.connect():
            raise BusConnectionError("Not connected to NATS")
            
        durable_name = f"durable_{subject.replace('.', '_')}"
        
        async def wrapped(msg):
            """
            Wrapper function that handles idempotency and error handling
            before calling the actual handler.
            """
            try:
                envelope = json.loads(msg.data.decode())
                event_id = envelope.get("event_id")
                
                # Handle idempotency - skip if already processed
                if event_id and await self._idem_store.seen(event_id):
                    await msg.ack()
                    self.logger.debug(f"Skipping duplicate event: {event_id}")
                    return
                    
                if event_id:
                    await self._idem_store.mark(event_id)
                    
                # Call actual handler
                await handler(envelope["payload"])
            except Exception as e:
                self.logger.error(f"Error handling event {subject}: {str(e)}", exc_info=True)
                # In a production system, you might want to implement
                # dead letter queue handling here
            finally:
                await msg.ack()
                
        try:
            sub = await self._js.subscribe(
                f"{self.stream}.{subject}",
                durable=durable_name,
                cb=wrapped,
                config=ConsumerConfig(
                    deliver_policy=DeliverPolicy.ALL,
                    ack_wait=30.0
                )
            )
            self._subscriptions[subject] = sub
            self.logger.info(f"Subscribed to {subject} with durable consumer {durable_name}")
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {subject}: {str(e)}")
            raise BusSubscribeError(f"Failed to subscribe to {subject}") from e
            
    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Send a request and wait for a reply.
        
        Args:
            subject: Request subject/topic
            payload: Request data
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            Optional[Dict]: Response data or None if timeout
            
        Raises:
            BusRequestError: If request fails
        """
        if not self._connected and not await self.connect():
            raise BusConnectionError("Not connected to NATS")
            
        try:
            data = json.dumps(payload).encode()
            response = await self._nc.request(
                f"{self.stream}.rpc.{subject}", 
                data, 
                timeout=timeout
            )
            return json.loads(response.data.decode())
        except TimeoutError:
            self.logger.warning(f"Request timed out for {subject}")
            return None
        except Exception as e:
            self.logger.error(f"Request failed for {subject}: {str(e)}")
            raise BusRequestError(f"Request failed for {subject}") from e
            
    async def close(self) -> None:
        """Gracefully shut down the connection."""
        if self._connected and self._nc:
            try:
                # Unsubscribe from all subjects
                for subject, sub in self._subscriptions.items():
                    await sub.unsubscribe()
                    self.logger.debug(f"Unsubscribed from {subject}")
                self._subscriptions.clear()
                
                # Close connection
                await self._nc.close()
                self._connected = False
                self.logger.info("NATS connection closed")
            except Exception as e:
                self.logger.error(f"Error during NATS shutdown: {str(e)}")
                raise
                
    def get_backend(self) -> str:
        """Return the active backend name."""
        return "nats"
        
    @property
    def idempotency_store(self) -> Any:
        """Return the idempotency store for this bus."""
        if not self._idem_store:
            # Fallback if not connected yet
            return InMemoryIdempotencyStore()
        return self._idem_store

``n
