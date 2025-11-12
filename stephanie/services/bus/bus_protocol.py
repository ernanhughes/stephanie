# stephanie/services/bus/bus_protocol.py
"""
Event Bus Protocol Definition (v2)

Unifies the interface across NATS / In-Process backends and adds readiness
and JetStream-style helpers with safe defaults.

Abstract (must implement):
- connect()
- publish(subject, payload, headers=None)
- subscribe(subject, handler, queue_group=None, **kwargs)
- request(subject, payload, timeout=5.0)
- close()
- get_backend()
- idempotency_store (property)

Optional helpers (default no-ops; backends may override):
- is_connected -> bool
- publish_raw(subject, body: bytes, headers=None) -> None
- wait_ready(timeout=5.0) -> bool
- ensure_stream(stream, subjects: list[str]) -> bool
- ensure_consumer(stream, subject, durable, **opts) -> bool
- flush(timeout=1.0) -> bool
- drain_subject(subject) -> bool
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Union
import base64


class BusProtocol(ABC):
    """Unified interface for all event bus implementations."""

    # ------------------ required (abstract) ------------------

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the bus backend. Return True on success."""
        raise NotImplementedError

    @abstractmethod
    async def publish(
        self,
        subject: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish a JSON-serializable payload to a subject."""
        raise NotImplementedError

    @abstractmethod
    async def subscribe(
        self,
        subject: str,
        handler: Callable[[Dict[str, Any]], None],
        queue_group: Optional[str] = None,
        **kwargs: Any,  # durable, ack_wait, max_deliver, deliver_policy, deliver_group, etc.
    ) -> None:
        """
        Subscribe to a subject. Implementations should accept common kwargs
        but may ignore those they don't support.
        """
        raise NotImplementedError

    @abstractmethod
    async def request(
        self,
        subject: str,
        payload: Dict[str, Any],
        timeout: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """Request/Reply on a subject. Return parsed JSON or None on timeout."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Gracefully close the connection, draining if possible."""
        raise NotImplementedError

    @abstractmethod
    def get_backend(self) -> str:
        """Return backend identifier (e.g., 'nats', 'inproc')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def idempotency_store(self) -> Any:
        """Return the idempotency store instance."""
        raise NotImplementedError

    # ------------------ optional (overridable) ------------------

    @property
    def is_connected(self) -> bool:
        """
        Indicate whether the backend is currently connected/usable.
        Backends should override for accuracy.
        """
        return True

    async def publish_raw(
        self,
        subject: str,
        body: bytes,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Optional fast path for raw bytes. Default wraps bytes into a JSON envelope
        so implementations that don't support raw can still deliver.
        """
        await self.publish(
            subject,
            {"__binary__": True, "data_b64": base64.b64encode(body).decode("ascii")},
            headers=headers,
        )

    async def wait_ready(self, timeout: float = 5.0) -> bool:
        """
        Optional readiness check. Backends can verify a health publish/flush.
        Default assumes ready once connected.
        """
        return True

    async def ensure_stream(self, stream: str, subjects: List[str]) -> bool:
        """
        Optional JS helper to ensure a stream and subjects exist.
        Default no-op returns True so callers can safely invoke this even on
        non-JetStream backends.
        """
        return True

    async def ensure_consumer(
        self,
        stream: str,
        subject: str,
        durable: str,
        *,
        ack_wait: Optional[Union[int, float, timedelta]] = None,
        max_deliver: Optional[int] = None,
        deliver_group: Optional[str] = None,
        deliver_policy: Optional[Any] = None,
    ) -> bool:
        """
        Optional JS helper to ensure a durable consumer exists.
        Default no-op returns True for compatibility on simpler backends.
        """
        return True

    async def flush(self, timeout: float = 1.0) -> bool:
        """
        Optional: wait for pending messages/acks. Default returns True.
        """
        return True

    async def drain_subject(self, subject: str) -> bool:
        """
        Optional: drain/purge a subject. Default returns True (no-op).
        """
        return True
