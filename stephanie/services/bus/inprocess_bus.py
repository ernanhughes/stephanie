# stephanie/services/bus/inprocess_bus.py
"""
In-process event bus for development and testing.
Provides a simple pub/sub implementation without external dependencies.
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
    """In-process event bus for development and testing."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._idem_store = InMemoryIdempotencyStore()
        self._connected = False
        self._loop = asyncio.get_event_loop()
        
    async def connect(self) -> bool:
        """Connect to the in-process bus (always succeeds)."""
        self._connected = True
        self.logger.info("Connected to in-process event bus (development mode)")
        return True
        
    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """Publish an event to all subscribers."""
        if not self._connected:
            if not await self.connect():
                return
                
        envelope = {
            "event_id": f"{subject}-{uuid.uuid4().hex}",
            "timestamp": time.time(),
            "subject": subject,
            "payload": payload
        }
        
        # Fire and forget for each subscriber
        if subject in self._subscribers:
            for handler in self._subscribers[subject]:
                asyncio.create_task(handler(envelope["payload"]))
                
    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to events on a subject."""
        if not self._connected:
            if not await self.connect():
                return
                
        if subject not in self._subscribers:
            self._subscribers[subject] = []
        self._subscribers[subject].append(handler)
        self.logger.debug(f"Subscribed to {subject}")
        
    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Send a request and wait for a reply (not supported in in-process)."""
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