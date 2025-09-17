# stephanie/services/bus/hybrid_bus.py
"""
Hybrid event bus that auto-selects the best available transport:
1. NATS JetStream (persistent, durable)
2. Redis Pub/Sub (transient)
3. In-process bus (dev-only)

All services use the same interface regardless of backend.
"""

import logging
from typing import Any, Callable, Dict, Optional

from .bus_protocol import BusProtocol
from .idempotency import InMemoryIdempotencyStore
from .inprocess_bus import InProcessKnowledgeBus
from .nats_bus import NatsKnowledgeBus


class HybridKnowledgeBus(BusProtocol):
    """Auto-selects the best available bus implementation."""
    
    def __init__(self, cfg: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self._bus: Optional[BusProtocol] = None
        self._idem_store = None
        self._backend = None
        
    async def connect(self) -> bool:
        """Connect to the best available bus backend."""
        # 1. Try NATS first (preferred for production)
        if self.cfg.get("bus", {}).get("backend") in [None, "nats"]:
            nats_bus = NatsKnowledgeBus(
                servers=self.cfg.get("bus", {}).get("servers", ["nats://localhost:4222"]),
                logger=self.logger
            )
            if await nats_bus.connect():
                self._bus = nats_bus
                self._backend = "nats"
                self._idem_store = nats_bus.idempotency_store
                return True
                
        # 2. Fall back to in-process (for development)
        inprocess_bus = InProcessKnowledgeBus(logger=self.logger)
        if await inprocess_bus.connect():
            self._bus = inprocess_bus
            self._backend = "inprocess"
            self._idem_store = inprocess_bus.idempotency_store
            return True
            
        self.logger.error("No bus backend available")
        return False
        
    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """Publish an event with standard envelope."""
        if not self._bus:
            if not await self.connect():
                return
                
        await self._bus.publish(subject, payload)
        
    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to events with idempotency handling."""
        if not self._bus:
            if not await self.connect():
                return
                
        await self._bus.subscribe(subject, handler)
        
    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Send a request and wait for a reply."""
        if not self._bus:
            if not await self.connect():
                return None
                
        return await self._bus.request(subject, payload, timeout)
        
    async def close(self) -> None:
        """Gracefully shut down the connection."""
        if self._bus:
            await self._bus.close()
            
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