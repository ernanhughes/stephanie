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

_logger = logging.getLogger(__name__)

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

    async def connect(self) -> bool:
        """
        Connect to the in-process bus.
        
        Returns:
            bool: Always returns True (in-process bus is always available)
        """
        self._connected = True
        _logger.debug("Connected to in-process event bus (development mode)")
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
            "payload": payload,
        }

        # 1) broadcast to non-group subscribers
        for handler in self._subscribers.get(subject, []):
            if asyncio.iscoroutinefunction(handler):
                asyncio.create_task(handler(envelope["payload"]))
            else:
                asyncio.create_task(self._loop.run_in_executor(None, handler, envelope["payload"]))

        # 2) queue-group delivery: one handler per group (round-robin)
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
        queue_group: Optional[str] = None,      # ← add
    ) -> None:
        """
        Subscribe to events on a subject.
        
        Args:
            subject: Event subject/topic to subscribe to
            handler: Callback function to handle events
        """
        if not self._connected:
            if not await self.connect():
                return

        if queue_group:
            key = (subject, queue_group)
            meta = self._qgroups.setdefault(key, {"handlers": [], "rr": 0})
            meta["handlers"].append(handler)
            self._groups_by_subject.setdefault(subject, []).append(queue_group) \
                if queue_group not in self._groups_by_subject.get(subject, []) else None
            _logger.debug(f"[inproc] Subscribed to {subject} in group '{queue_group}' "
                          f"({len(meta['handlers'])} handlers in group)")
            return

        self._subscribers.setdefault(subject, []).append(handler)
        _logger.debug(f"[inproc] Subscribed to {subject} "
                      f"({len(self._subscribers[subject])} non-group handlers)")

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
        _logger.warning("Request/reply pattern not supported in in-process bus")
        return None
        
    async def close(self) -> None:
        """Gracefully shut down the connection."""
        self._subscribers.clear()
        self._connected = False
        _logger.debug("In-process bus disconnected")
        
    def get_backend(self) -> str:
        """Return the active backend name."""
        return "inprocess"
        
    @property
    def idempotency_store(self):
        """Access the idempotency store for this bus."""
        return self._idem_store