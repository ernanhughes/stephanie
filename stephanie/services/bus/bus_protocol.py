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
    async def publish(self, subject: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> None:
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
    async def subscribe(
        self,
        subject: str,
        handler: Callable[[Dict[str, Any]], None],
        queue_group: Optional[str] = None,
    ) -> None:
        """
        Subscribe to messages on the specified subject.

        Args:
            subject: Topic/channel to subscribe to
            handler: Callback function to handle incoming messages
            queue_group: Optional queue group for load balancing (if supported by backend)

        Note:
            Handler should be designed to process messages idempotently
        """
        pass

    @abstractmethod
    async def request(
        self, subject: str, payload: Dict[str, Any], timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
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
