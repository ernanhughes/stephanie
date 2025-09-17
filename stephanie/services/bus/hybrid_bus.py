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

