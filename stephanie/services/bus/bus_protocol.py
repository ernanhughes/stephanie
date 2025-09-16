# stephanie/services/bus/bus_protocol.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class BusProtocol(ABC):
    """Unified interface for all event bus implementations."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the bus backend. Returns True if successful."""
        pass
        
    @abstractmethod
    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """Publish a message to the specified subject."""
        pass
        
    @abstractmethod
    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to messages on the specified subject."""
        pass
        
    @abstractmethod
    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Send a request and wait for a reply."""
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """Gracefully shut down the connection."""
        pass
        
    @abstractmethod
    def get_backend(self) -> str:
        """Return the name of the active backend."""
        pass
        
    @property
    @abstractmethod
    def idempotency_store(self) -> Any:
        """Return the idempotency store for this bus."""
        pass