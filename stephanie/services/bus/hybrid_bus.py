from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from .bus_protocol import BusProtocol
from .errors import (BusConnectionError, BusPublishError, BusRequestError,
                     BusSubscribeError)
from .idempotency import InMemoryIdempotencyStore
from .inprocess_bus import InProcessKnowledgeBus
from .nats_bus import NatsKnowledgeBus

_logger = logging.getLogger(__name__)


class HybridKnowledgeBus(BusProtocol):
    """
    Flat-config only.

    Expected cfg shape (flat dict):
      {
        "enabled": true,
        "backend": "nats",                # "nats" | "inproc"
        "servers": "nats://localhost:4222" or ["nats://..."],
        "stream": "stephanie",
        "required": false,                # or "strict": false
        "strict": false,
        "connect_timeout_s": 2.0,
        "fallback": "inproc"              # "inproc" | "none"
      }
    """

    def __init__(self, cfg: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.cfg = dict(cfg or {})  # flat only
        self.logger = logger or _logger
        self._bus: Optional[BusProtocol] = None
        self._backend: str = "none"
        self._idem_store = None
        self._disabled = False

    # --------------- helpers ---------------

    def _norm_servers(self, servers: Any) -> List[str]:
        if servers is None:
            return ["nats://localhost:4222"]
        if isinstance(servers, str):
            return [servers]
        if isinstance(servers, (list, tuple)):
            return [str(s) for s in servers]
        return ["nats://localhost:4222"]

    def _norm(self) -> Dict[str, Any]:
        enabled = bool(self.cfg.get("enabled", True))
        required = bool(self.cfg.get("required", False) or self.cfg.get("strict", False))
        return {
            "enabled": enabled,
            "backend": (self.cfg.get("backend") or "nats").lower(),
            "servers": self._norm_servers(self.cfg.get("servers")),
            "stream": self.cfg.get("stream", "stephanie"),
            "required": required,
            "timeout": float(self.cfg.get("connect_timeout_s", 2.0)),
            "fallback": (self.cfg.get("fallback", "inproc") or "inproc").lower(),
        }

    async def _with_timeout(self, coro, timeout: float) -> Any:
        return await asyncio.wait_for(coro, timeout=timeout)

    # --------------- connect/fallback ---------------

    async def connect(self, *, timeout: Optional[float] = None) -> bool:
        cfg = self._norm()

        if not cfg["enabled"]:
            self._disabled = True
            self._bus = None
            self._backend = "none"
            self._idem_store = None
            self.logger.info("Hybrid bus disabled by config; continuing without bus.")
            return True  # disabled is not an error

        if self._bus is not None:
            return True  # already connected

        timeout = cfg["timeout"] if timeout is None else float(timeout)

        # Try NATS first
        if cfg["backend"] in ("nats", None):
            try:
                nats_bus = NatsKnowledgeBus(
                    servers=cfg["servers"],
                    stream=cfg["stream"],
                    logger=self.logger,
                )
                ok = await self._with_timeout(nats_bus.connect(), timeout)
                if ok:
                    self._bus = nats_bus
                    self._backend = "nats"
                    self._idem_store = getattr(nats_bus, "idempotency_store", None)
                    self.logger.info("Connected to NATS JetStream bus")
                    return True
            except asyncio.TimeoutError:
                self.logger.warning(f"NATS connection timed out (<= {timeout}s).")
            except Exception as e:
                self.logger.warning("NATS connection failed: %r", e)
            if cfg["required"]:
                raise BusConnectionError("NATS connection required but unavailable")

        # Fallbacks
        if cfg["fallback"] == "inproc":
            try:
                inproc = InProcessKnowledgeBus(logger=self.logger)
                ok = await self._with_timeout(inproc.connect(), timeout)
                if ok:
                    self._bus = inproc
                    self._backend = "inproc"
                    self._idem_store = getattr(inproc, "idempotency_store", None)
                    self.logger.info("Connected to in-process event bus (fallback).")
                    return True
            except asyncio.TimeoutError:
                self.logger.warning(f"InProcessBus connect timed out (<= {timeout}s).")
            except Exception as e:
                self.logger.error(f"InProcessBus connect error: {e}")

        self._bus = None
        self._backend = "none"
        self._idem_store = None
        self.logger.error("No bus backend available (nats down, no usable fallback).")
        return False

    # --------------- API ---------------

    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        if self._bus is None and not self._disabled:
            ok = await self.connect()
            if not ok:
                raise BusConnectionError("No bus connection available")
        if self._bus is None:  # disabled
            return
        try:
            await self._bus.publish(subject, payload)
        except Exception as e:
            self.logger.error(f"Failed to publish to {subject}: {e}")
            raise BusPublishError(f"Failed to publish to {subject}") from e

    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        if self._bus is None and not self._disabled:
            ok = await self.connect()
            if not ok:
                raise BusConnectionError("No bus connection available")
        if self._bus is None:  # disabled
            return
        try:
            await self._bus.subscribe(subject, handler)
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {subject}: {e}")
            raise BusSubscribeError(f"Failed to subscribe to {subject}") from e

    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        if self._bus is None and not self._disabled:
            ok = await self.connect()
            if not ok:
                raise BusConnectionError("No bus connection available")
        if self._bus is None:  # disabled
            return None
        try:
            return await self._bus.request(subject, payload, timeout)
        except Exception as e:
            self.logger.error(f"Request failed for {subject}: {e}")
            raise BusRequestError(f"Request failed for {subject}") from e

    async def close(self) -> None:
        if self._bus:
            try:
                await self._bus.close()
                self.logger.info("Bus connection closed")
            except Exception as e:
                self.logger.error(f"Error during bus shutdown: {e}")
        self._bus = None
        self._backend = "none"
        self._idem_store = None

    def get_backend(self) -> str:
        return self._backend

    @property
    def idempotency_store(self):
        return self._idem_store or InMemoryIdempotencyStore()
