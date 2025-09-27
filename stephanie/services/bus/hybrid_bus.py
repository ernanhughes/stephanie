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
from .errors import (BusConnectionError, BusPublishError, BusRequestError,
                     BusSubscribeError)
from .idempotency import InMemoryIdempotencyStore
from .inprocess_bus import InProcessKnowledgeBus
from .nats_bus import \
    NatsKnowledgeBus  # <-- OK now; nats_bus no longer imports hybrid_bus

_logger = logging.getLogger(__name__)

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
        self._backend: Optional[str] = None

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
        # Accept both {"bus": {...}} and {...}
        bus_config = self.cfg.get("bus", None)
        if bus_config is None and isinstance(self.cfg, dict) and "backend" in self.cfg:
            bus_config = self.cfg  # flat shape accepted
        if bus_config is None:
            bus_config = {}        # final fallback

        preferred_backend = bus_config.get("backend")

        # NATS first...
        if preferred_backend in (None, "nats"):
            try:
                nats_bus = NatsKnowledgeBus(
                    servers=bus_config.get("servers", ["nats://localhost:4222"]),
                    stream=bus_config.get("stream", "stephanie"),
                    logger=self.logger,
                )
                if await nats_bus.connect():
                    self._bus = nats_bus
                    self._backend = "nats"
                    self._idem_store = nats_bus.idempotency_store
                    _logger.info("Connected to NATS JetStream bus")
                    return True
            except Exception as e:
                _logger.warning(f"NATS connection failed: {e}")

        # (optionally) Redis here...

        # In-process fallback
        try:
            inproc = InProcessKnowledgeBus(logger=self.logger)
            if await inproc.connect():
                self._bus = inproc
                self._backend = "inprocess"
                self._idem_store = inproc.idempotency_store
                _logger.info("Connected to in-process event bus (dev mode)")
                return True
        except Exception as e:
            _logger.error(f"In-process bus failed: {e}")

        _logger.error("No bus backend available")
        return False

    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """Publish an event with standard envelope."""
        if not self._bus and not await self.connect():
            raise BusConnectionError("No bus connection available")
        try:
            await self._bus.publish(subject, payload)
        except Exception as e:
            _logger.error(f"Failed to publish to {subject}: {e}")
            raise BusPublishError(f"Failed to publish to {subject}") from e

    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to events with idempotency handling."""
        if not self._bus and not await self.connect():
            raise BusConnectionError("No bus connection available")
        try:
            await self._bus.subscribe(subject, handler)
        except Exception as e:
            _logger.error(f"Failed to subscribe to {subject}: {e}")
            raise BusSubscribeError(f"Failed to subscribe to {subject}") from e

    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Send a request and wait for a reply."""
        if not self._bus and not await self.connect():
            raise BusConnectionError("No bus connection available")
        try:
            return await self._bus.request(subject, payload, timeout)
        except Exception as e:
            _logger.error(f"Request failed for {subject}: {e}")
            raise BusRequestError(f"Request failed for {subject}") from e

    async def close(self) -> None:
        """<<< This was missing, causing the ABC error >>>"""
        if self._bus:
            try:
                await self._bus.close()
                _logger.info("Bus connection closed")
            except Exception as e:
                _logger.error(f"Error during bus shutdown: {e}")

    def get_backend(self) -> str:
        return self._backend or "none"

    @property
    def idempotency_store(self):
        return self._idem_store or InMemoryIdempotencyStore()
