# stephanie/services/bus/hybrid_bus.py
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
        self.logger = logger
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
            _logger.info("Hybrid bus disabled by config; continuing without bus.")
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
                    _logger.info("Connected to NATS JetStream bus")
                    return True
            except asyncio.TimeoutError:
                _logger.warning(f"NATS connection timed out (<= {timeout}s).")
            except Exception as e:
                _logger.warning("NATS connection failed: %r", e)
            if cfg["required"]:
                raise BusConnectionError("NATS connection required but unavailable")

        # Fallbacks
        if cfg["fallback"] == "inproc":
            try:
                inproc = InProcessKnowledgeBus(logger=_logger)
                ok = await self._with_timeout(inproc.connect(), timeout)
                if ok:
                    self._bus = inproc
                    self._backend = "inproc"
                    self._idem_store = getattr(inproc, "idempotency_store", None)
                    _logger.info("Connected to in-process event bus (fallback).")
                    return True
            except asyncio.TimeoutError:
                _logger.warning(f"InProcessBus connect timed out (<= {timeout}s).")
            except Exception as e:
                _logger.error(f"InProcessBus connect error: {e}")

        self._bus = None
        self._backend = "none"
        self._idem_store = None
        _logger.error("No bus backend available (nats down, no usable fallback).")
        return False

    async def close(self):
        async with self._lock:
            if self._nc:
                try:
                    await self._nc.close()
                except Exception:
                    pass
            self._nc = None
            self._js = None
            self._connected = False
    # --------------- API ---------------

    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        _logger.debug(f"Publishing to {subject}: {payload}")
        if self._bus is None and not self._disabled:
            ok = await self.connect()
            if not ok:
                raise BusConnectionError("No bus connection available")
        if self._bus is None:  # disabled
            return
        try:
            await self._bus.publish(subject, payload)
        except Exception as e:
            _logger.error(f"Failed to publish to {subject}: {e}")
            raise BusPublishError(f"Failed to publish to {subject}") from e

    async def subscribe(
        self,
        subject: str,
        handler: Callable[[Dict[str, Any]], None],
    ) -> None:
        if self._bus is None and not self._disabled:
            ok = await self.connect()
            if not ok:
                raise BusConnectionError("No bus connection available")
        if self._bus is None:
            return
        try:
            return await self._bus.subscribe(subject, handler)
        except Exception as e:
            _logger.error(f"Failed to subscribe to {subject}: {e}")
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
            _logger.error(f"Request failed for {subject}: {e}")
            raise BusRequestError(f"Request failed for {subject}") from e

    def health_check(self) -> dict:
        """Returns detailed health status including bus type and connection state"""
        if self._disabled:
            return {
                "is_healthy": True,
                "bus_type": "disabled",
                "status": "disabled",
                "backend": {self._backend},
                "details": "Bus explicitly disabled by config"
            }
        
        if self._bus is None:
            return {
                "is_healthy": False,
                "bus_type": "none",
                "backend": {self._backend},
                "status": "not_connected",
                "details": "No bus instance created"
            }

        try:
            if self._backend == "nats":
                # Check NATS-specific connection state
                nats_connected = getattr(self._bus, '_connected', False)
                nats_closed = getattr(self._bus, 'is_closed', True)
                
                # Check connection details
                conn_details = {}
                if hasattr(self._bus, 'debug_connection_status'):
                    conn_details = self._bus.debug_connection_status()
                
                is_healthy = nats_connected and not nats_closed
                return {
                    "is_healthy": is_healthy,
                    "bus_type": "nats",
                    "status": "connected" if nats_connected else "disconnected",
                    "details": {
                        "connected": nats_connected,
                        "closed": nats_closed,
                        "servers": self._bus.servers,
                        "stream": self._bus.stream,
                        "connection_uptime": conn_details.get("connection_uptime", 0),
                        "reconnect_attempts": conn_details.get("reconnect_attempts", 0),
                        "debug_mode": conn_details.get("debug_mode", False)
                    }
                }
            
            elif self._backend == "inproc":
                # InProcessBus always connected if initialized
                return {
                    "is_healthy": True,
                    "bus_type": "inproc",
                    "status": "connected",
                    "details": {
                        "subscriptions": len(self._bus._subscribers),
                        "idempotency": bool(self._bus.idempotency_store),
                        "memory_usage": f"{id(self._bus):x}"
                    }
                }
            
            else:
                return {
                    "is_healthy": False,
                    "bus_type": self._backend,
                    "status": "unsupported",
                    "details": f"Unsupported bus type: {self._backend}"
                }
                
        except Exception as e:
            return {
                "is_healthy": False,
                "bus_type": self._backend,
                "status": "error",
                "details": f"Health check failed: {str(e)}"
            }

    async def close(self) -> None:
        if self._bus:
            try:
                await self._bus.close()
                _logger.info("Bus connection closed")
            except Exception as e:
                _logger.error(f"Error during bus shutdown: {e}")
        self._bus = None
        self._backend = "none"
        self._idem_store = None

    def get_backend(self) -> str:
        return self._backend

    @property
    def idempotency_store(self):
        return self._idem_store or InMemoryIdempotencyStore()
