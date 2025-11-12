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
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from .bus_protocol import BusProtocol
from .idempotency import InMemoryIdempotencyStore

log = logging.getLogger(__name__)

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
        self.logger = logger
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._idem_store = InMemoryIdempotencyStore()
        self._connected = False
        self._loop = asyncio.get_event_loop()
        self._qgroups: Dict[tuple, Dict[str, Any]] = {}             # (subject, group) -> {"handlers":[...], "rr":0}
        self._groups_by_subject: Dict[str, List[str]] = {}           # subject -> [group,...]
        self._idem_store = InMemoryIdempotencyStore()
        self._connected = False
        self._loop = asyncio.get_event_loop()

       
    
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
        log.warning("Request/reply pattern not supported in in-process bus")
        return None
        
    async def close(self) -> None:
        """Gracefully shut down the connection."""
        self._subscribers.clear()
        self._connected = False
        log.debug("In-process bus disconnected")
        
    def get_backend(self) -> str:
        """Return the active backend name."""
        return "inprocess"
        
    # ---------- Compatibility helpers (no-op / emulated) ----------


    @property
    def is_connected(self) -> bool:
        return bool(self._connected)

    async def connect(self) -> bool:
        self._connected = True
        log.debug("[inproc] connected")
        return True

    async def publish(self, subject: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> None:
        # headers accepted for parity, ignored
        if not self._connected:
            if not await self.connect():
                return
        envelope = {
            "event_id": f"{subject}-{uuid.uuid4().hex}",
            "timestamp": time.time(),
            "subject": subject,
            "payload": payload,
        }
        # broadcast to non-group handlers
        for handler in self._subscribers.get(subject, []):
            if asyncio.iscoroutinefunction(handler):
                asyncio.create_task(handler(envelope["payload"]))
            else:
                asyncio.create_task(self._loop.run_in_executor(None, handler, envelope["payload"]))
        # group (round-robin)
        for group in self._groups_by_subject.get(subject, []):
            key = (subject, group)
            meta = self._qgroups.get(key)
            if not meta or not meta["handlers"]:
                continue
            idx = meta["rr"] % len(meta["handlers"])
            meta["rr"] = (meta["rr"] + 1) % max(1, len(meta["handlers"]))
            handler = meta["handlers"][idx]
            if asyncio.iscoroutinefunction(handler):
                asyncio.create_task(handler(envelope["payload"]))
            else:
                asyncio.create_task(self._loop.run_in_executor(None, handler, envelope["payload"]))

    async def subscribe(
        self,
        subject: str,
        handler: Callable[[Dict[str, Any]], None],
        *,
        queue: Optional[str] = None,
        queue_group: Optional[str] = None,
        deliver_group: Optional[str] = None,
        durable: Optional[str] = None,
        ack_wait: Optional[int] = None,
        max_deliver: Optional[int] = None,
        deliver_policy: Optional[Any] = None,
    ) -> None:
        # Parity: accept same kwargs as NATS; most are ignored here.
        if not self._connected:
            if not await self.connect():
                return
        qgroup = queue or deliver_group or queue_group
        if qgroup:
            key = (subject, qgroup)
            meta = self._qgroups.setdefault(key, {"handlers": [], "rr": 0})
            meta["handlers"].append(handler)
            if qgroup not in self._groups_by_subject.get(subject, []):
                self._groups_by_subject.setdefault(subject, []).append(qgroup)
            log.debug("[inproc] Subscribed %s in group '%s' (%d handlers)",
                      subject, qgroup, len(meta["handlers"]))
            return
        self._subscribers.setdefault(subject, []).append(handler)
        log.debug("[inproc] Subscribed %s (%d non-group handlers)",
                  subject, len(self._subscribers[subject]))

    async def wait_ready(self, timeout: float = 5.0) -> bool:
        """Always ready once connected; returns True."""
        if not self._connected:
            await self.connect()
        log.debug("[inproc] wait_ready -> True")
        return True

    async def ensure_stream(self, stream: str, subjects: List[str]) -> bool:
        """
        Emulate JetStream ensure_stream: we just record subjects to help with debugging.
        """
        if not self._connected:
            await self.connect()
        if not hasattr(self, "_streams"):
            self._streams: Dict[str, set] = {}
        s = self._streams.setdefault(stream, set())
        before = len(s)
        for sub in subjects or []:
            s.add(str(sub))
        after = len(s)
        log.debug("[inproc] ensure_stream stream=%s subjects_now=%d (changed=%s)", stream, after, after != before)
        return True

    async def ensure_consumer(
        self,
        stream: str,
        subject: str,
        durable: str,
        *,
        ack_wait: Optional[int] = None,
        max_deliver: Optional[int] = None,
        deliver_group: Optional[str] = None,
        deliver_policy: Optional[Any] = None,
    ) -> bool:
        """
        Emulate consumer creation: record durable and subject mapping.
        """
        if not self._connected:
            await self.connect()
        if not hasattr(self, "_consumers"):
            self._consumers: Dict[str, Dict[str, Any]] = {}
        key = f"{stream}:{durable}"
        self._consumers[key] = {
            "stream": stream,
            "subject": subject,
            "durable": durable,
            "ack_wait": ack_wait,
            "max_deliver": max_deliver,
            "deliver_group": deliver_group,
        }
        log.debug("[inproc] ensure_consumer stream=%s durable=%s subject=%s group=%s", stream, durable, subject, deliver_group)
        return True


    @property
    def idempotency_store(self):
        """Access the idempotency store for this bus."""
        return self._idem_store