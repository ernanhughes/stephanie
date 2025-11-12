# stephanie/services/bus/hybrid_bus.py
from __future__ import annotations

import asyncio
import base64
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from .bus_protocol import BusProtocol
from .idempotency import InMemoryIdempotencyStore
from .inprocess_bus import InProcessKnowledgeBus
from .nats_bus import NatsKnowledgeBus

log = logging.getLogger(__name__)


class HybridKnowledgeBus(BusProtocol):
    """
    Flat-config only.

    cfg example (flat dict):
      {
        "enabled": true,
        "backend": "nats",                # "nats" | "inproc" | "none"
        "servers": "nats://localhost:4222" or ["nats://..."],
        "stream": "stephanie",
        "required": false,                # if true and nats fails, do NOT fallback (but we still won't raise on first-use ops)
        "strict": false,                  # alias for required
        "connect_timeout_s": 2.0,
        "fallback": "inproc"              # "inproc" | "none"
      }
    """

    def __init__(self, cfg: Dict[str, Any], logger: Optional[logging.Logger] = None):
        # store flat config; do NOT connect here
        self.cfg = dict(cfg or {})
        self.logger = logger or log

        self._bus: Optional[BusProtocol] = None
        self._backend: str = "none"           # "nats" | "inproc" | "none"
        self._disabled: bool = False
        self._idem_store: Optional[InMemoryIdempotencyStore] = None

        # single-flight guard for concurrent first-use connects
        self._connect_lock = asyncio.Lock()

    # ---------------------- helpers ----------------------

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
            "backend": (self.cfg.get("backend") or "nats").lower(),   # nats|inproc|none
            "servers": self._norm_servers(self.cfg.get("servers")),
            "stream": self.cfg.get("stream", "stephanie"),
            "required": required,
            "timeout": float(self.cfg.get("connect_timeout_s", 2.0)),
            "fallback": (self.cfg.get("fallback", "inproc") or "inproc").lower(),  # inproc|none
        }

    async def _with_timeout(self, coro, timeout: float) -> Any:
        return await asyncio.wait_for(coro, timeout=timeout)

    @property
    def is_connected(self) -> bool:
        if not self._bus:
            return False
        # Prefer backend's own flag if present
        return bool(getattr(self._bus, "is_connected", True))

    def get_backend(self) -> str:
        return self._backend

    # ---------------------- connect / fallback ----------------------

    async def connect(self, *, timeout: Optional[float] = None) -> bool:
        """
        Best-effort connect. Never raises. Returns True on any active backend.
        Leaves self._bus/_backend set appropriately.
        """
        cfg = self._norm()

        if not cfg["enabled"]:
            self._disabled = True
            self._bus = None
            self._backend = "none"
            self._idem_store = None
            log.debug("Hybrid bus disabled by config; continuing without bus.")
            return True  # disabled-by-config is not an error

        if self._bus is not None:
            return True  # already connected/active

        timeout = cfg["timeout"] if timeout is None else float(timeout)

        # Try backend choice
        if cfg["backend"] == "nats":
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
                    log.info("Hybrid bus connected to NATS JetStream.")
                    return True
            except asyncio.TimeoutError:
                log.warning("Hybrid bus: NATS connection timed out (<= %ss).", timeout)
            except Exception as e:
                log.warning("Hybrid bus: NATS connection failed: %r", e)

            # If required, we won't force-raise here—first-use ops should remain non-fatal.
            # We'll just avoid swapping to fallback if required==True.
            if not cfg["required"]:
                # fall back if allowed
                if cfg["fallback"] == "inproc":
                    try:
                        inproc = InProcessKnowledgeBus(logger=self.logger)
                        ok = await self._with_timeout(inproc.connect(), timeout)
                        if ok:
                            self._bus = inproc
                            self._backend = "inproc"
                            self._idem_store = getattr(inproc, "idempotency_store", None)
                            log.debug("Hybrid bus using in-process fallback (NATS unavailable).")
                            return True
                    except Exception as e:
                        log.error("Hybrid bus: InProc fallback connect error: %s", e)

            # No usable backend
            self._bus = None
            self._backend = "none"
            self._idem_store = None
            log.error("Hybrid bus: no backend available (NATS down, fallback disabled).")
            return False

        elif cfg["backend"] == "inproc":
            try:
                inproc = InProcessKnowledgeBus(logger=self.logger)
                ok = await self._with_timeout(inproc.connect(), timeout)
                if ok:
                    self._bus = inproc
                    self._backend = "inproc"
                    self._idem_store = getattr(inproc, "idempotency_store", None)
                    log.info("Hybrid bus using in-process backend.")
                    return True
            except Exception as e:
                log.error("Hybrid bus: InProc connect error: %s", e)
            self._bus = None
            self._backend = "none"
            self._idem_store = None
            return False

        else:
            # backend == "none"
            self._bus = None
            self._backend = "none"
            self._idem_store = None
            return False

    async def _ensure_connected_for_use(self) -> bool:
        """
        Single-flight guard that attempts a best-effort connect on first use.
        Never raises; returns True if some backend is active, False otherwise.
        """
        # already active?
        if self._bus is not None:
            return True
        # disabled?
        if self._disabled:
            return False

        async with self._connect_lock:
            if self._bus is not None:
                return True
            ok = await self.connect()
            return ok

    # ---------------------- Public API ----------------------

    async def publish(self, subject: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> None:
        # First-use connect attempt; non-fatal
        if not await self._ensure_connected_for_use():
            log.debug("Hybrid bus: publish dropped (no backend): %s", subject)
            return

        try:
            # Prefer backend signature with headers if supported
            sig = inspect.signature(self._bus.publish)  # type: ignore[attr-defined]
            if "headers" in sig.parameters:
                await self._bus.publish(subject, payload, headers=headers)  # type: ignore[misc]
            else:
                await self._bus.publish(subject, payload)                  # type: ignore[misc]
        except Exception as e:
            # If the active backend died mid-flight, try one soft reconnect + retry
            log.warning("Hybrid bus: publish error on %s: %s (retrying once)", subject, e)
            self._bus = None
            self._backend = "none"
            if await self._ensure_connected_for_use():
                try:
                    sig = inspect.signature(self._bus.publish)  # type: ignore[attr-defined]
                    if "headers" in sig.parameters:
                        await self._bus.publish(subject, payload, headers=headers)  # type: ignore[misc]
                    else:
                        await self._bus.publish(subject, payload)                  # type: ignore[misc]
                    return
                except Exception as e2:
                    log.error("Hybrid bus: publish retry failed on %s: %s", subject, e2)
            # Final: drop silently (your requirement)
            # If you prefer visibility, switch to raise BusPublishError here.
            return

    async def publish_raw(self, subject: str, body: bytes, headers: Optional[Dict[str, str]] = None) -> None:
        if not await self._ensure_connected_for_use():
            log.debug("Hybrid bus: publish_raw dropped (no backend): %s", subject)
            return

        # If backend supports publish_raw, use it; else wrap bytes in a base64 envelope
        try:
            if hasattr(self._bus, "publish_raw"):
                sig = inspect.signature(getattr(self._bus, "publish_raw"))  # type: ignore
                if "headers" in sig.parameters:
                    await getattr(self._bus, "publish_raw")(subject, body, headers=headers)  # type: ignore[misc]
                else:
                    await getattr(self._bus, "publish_raw")(subject, body)                   # type: ignore[misc]
                return
        except Exception as e:
            log.warning("Hybrid bus: publish_raw direct path failed: %s (will fallback)", e)

        env = {"__binary__": True, "data_b64": base64.b64encode(body).decode("ascii")}
        await self.publish(subject, env, headers=headers)

    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], None], **kwargs: Any) -> None:
        if not await self._ensure_connected_for_use():
            log.debug("Hybrid bus: subscribe ignored (no backend): %s", subject)
            return

        # normalize & filter kwargs to what backend supports
        norm = dict(kwargs)
        if "queue_group" in norm and "deliver_group" not in norm:
            norm["deliver_group"] = norm.pop("queue_group")
        if "group" in norm and "deliver_group" not in norm:
            norm["deliver_group"] = norm.pop("group")

        try:
            sig = inspect.signature(self._bus.subscribe)  # type: ignore[attr-defined]
            filtered = {k: v for k, v in norm.items() if k in sig.parameters}
        except Exception:
            filtered = norm

        try:
            await self._bus.subscribe(subject, handler, **filtered)  # type: ignore[misc]
        except Exception as e:
            # Same soft-reconnect approach as publish
            log.warning("Hybrid bus: subscribe error on %s: %s (retrying once)", subject, e)
            self._bus = None
            self._backend = "none"
            if await self._ensure_connected_for_use():
                try:
                    await self._bus.subscribe(subject, handler, **filtered)  # type: ignore[misc]
                except Exception as e2:
                    log.error("Hybrid bus: subscribe retry failed on %s: %s", subject, e2)
            return

    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        if not await self._ensure_connected_for_use():
            log.debug("Hybrid bus: request short-circuited (no backend): %s", subject)
            return None
        try:
            return await self._bus.request(subject, payload, timeout)  # type: ignore[misc]
        except Exception as e:
            log.warning("Hybrid bus: request error on %s: %s (retrying once)", subject, e)
            self._bus = None
            self._backend = "none"
            if await self._ensure_connected_for_use():
                try:
                    return await self._bus.request(subject, payload, timeout)  # type: ignore[misc]
                except Exception as e2:
                    log.error("Hybrid bus: request retry failed on %s: %s", subject, e2)
            return None  # non-fatal

    def health_check(self) -> dict:
        if self._disabled:
            return {
                "is_healthy": True,
                "bus_type": "disabled",
                "backend": self._backend,
                "status": "disabled",
                "details": "Bus explicitly disabled by config",
            }

        if self._bus is None:
            return {
                "is_healthy": False,
                "bus_type": "none",
                "backend": self._backend,
                "status": "not_connected",
                "details": "No bus instance created",
            }

        try:
            if self._backend == "nats":
                nats_connected = bool(getattr(self._bus, "_connected", False))
                nats_closed = bool(getattr(self._bus, "is_closed", True))
                conn_details = {}
                if hasattr(self._bus, "debug_connection_status"):
                    conn_details = self._bus.debug_connection_status()

                return {
                    "is_healthy": nats_connected and not nats_closed,
                    "bus_type": "nats",
                    "backend": "nats",
                    "status": "connected" if nats_connected else "disconnected",
                    "details": {
                        "connected": nats_connected,
                        "closed": nats_closed,
                        "servers": getattr(self._bus, "servers", []),
                        "stream": getattr(self._bus, "stream", ""),
                        "connection_uptime": conn_details.get("connection_uptime", 0),
                        "reconnect_attempts": conn_details.get("reconnect_attempts", 0),
                        "debug_mode": conn_details.get("debug_mode", False),
                    },
                }

            if self._backend == "inproc":
                return {
                    "is_healthy": True,
                    "bus_type": "inproc",
                    "backend": "inproc",
                    "status": "connected",
                    "details": {
                        "subscriptions": len(getattr(self._bus, "_subscribers", [])),
                        "idempotency": bool(getattr(self._bus, "idempotency_store", None)),
                    },
                }

            return {
                "is_healthy": False,
                "bus_type": self._backend,
                "backend": self._backend,
                "status": "unsupported",
                "details": f"Unsupported bus type: {self._backend}",
            }

        except Exception as e:
            return {
                "is_healthy": False,
                "bus_type": self._backend,
                "backend": self._backend,
                "status": "error",
                "details": f"Health check failed: {str(e)}",
            }

    async def close(self) -> None:
        if self._bus:
            try:
                await self._bus.close()
                log.debug("Hybrid bus connection closed.")
            except Exception as e:
                log.error(f"Hybrid bus: error during shutdown: {e}")
        self._bus = None
        self._backend = "none"
        self._idem_store = None


    # ---------------------- Stream/Consumer helpers ----------------------

    async def wait_ready(self, timeout: float = 5.0) -> bool:
        log.debug("[HybridBus] wait_ready(start) timeout=%.2fs", timeout)
        if not await self._ensure_connected_for_use():
            log.warning("[HybridBus] wait_ready: no backend available")
            return False
        try:
            ok = await self._bus.wait_ready(timeout)  # type: ignore[attr-defined]
            log.info("[HybridBus] wait_ready -> %s (backend=%s)", ok, self._backend)
            return bool(ok)
        except Exception as e:
            log.warning("[HybridBus] wait_ready delegate failed: %s", e)
        # If backend lacks the method, consider the active connection as "ready"
        log.info("[HybridBus] wait_ready: assuming ready (backend=%s)", self._backend)
        return True

    async def ensure_stream(self, stream: str, subjects: List[str]) -> bool:
        if not await self._ensure_connected_for_use():
            log.warning("[HybridBus] ensure_stream dropped (no backend): stream=%s", stream)
            return False
        log.debug("[HybridBus] ensure_stream stream=%s subjects=%s backend=%s", stream, subjects, self._backend)
        try:
            ok = await self._bus.ensure_stream(stream, subjects)  # type: ignore[attr-defined]
            log.debug("[HybridBus] ensure_stream delegated -> %s", ok)
            return bool(ok)
        except Exception as e:
            log.warning("[HybridBus] ensure_stream delegate failed: %s", e)
        # If backend does not support it, treat as success (no-op)
        log.warning("[HybridBus] ensure_stream not supported by backend=%s; continuing", self._backend)
        return True

    async def ensure_consumer(
        self,
        stream: str,
        subject: str,
        durable: str,
        *,
        ack_wait: Optional[int] = None,
        max_deliver: Optional[int] = None,
        deliver_group: Optional[str] = None,
        deliver_policy: Optional[Any] = None,
    ) -> bool:
        if not await self._ensure_connected_for_use():
            log.warning("[HybridBus] ensure_consumer dropped (no backend): stream=%s durable=%s", stream, durable)
            return False
        log.debug("[HybridBus] ensure_consumer stream=%s subject=%s durable=%s backend=%s",
                          stream, subject, durable, self._backend)
        try:
            ok = await self._bus.ensure_consumer(
                stream=stream,
                subject=subject,
                durable=durable,
                ack_wait=ack_wait,
                max_deliver=max_deliver,
                deliver_group=deliver_group,
                deliver_policy=deliver_policy,
            )  # type: ignore[attr-defined]
            log.info("[HybridBus] ensure_consumer delegated -> %s", ok)
            return bool(ok)
        except Exception as e:
            log.warning("[HybridBus] ensure_consumer delegate failed: %s", e)
        # If backend does not support it, treat as success (no-op)
        log.warning("[HybridBus] ensure_consumer not supported by backend=%s; continuing", self._backend)
        return True

    # --------------------- flush / drain passthrough ---------------------

    async def flush(self, timeout: float = 1.0) -> bool:
        if not self._bus:
            return False
        try:
            if hasattr(self._bus, "flush"):
                return await self._bus.flush(timeout=timeout)
        except Exception as e:
            log.warning(f"Hybrid bus flush failed: {e}")
        return False

    async def drain_subject(self, subject: str) -> bool:
        if not self._bus:
            return False
        try:
            if hasattr(self._bus, "drain_subject"):
                return await self._bus.drain_subject(subject)
        except Exception as e:
            log.warning(f"Hybrid bus drain_subject failed: {e}")
        return False

    @property
    def idempotency_store(self):
        return self._idem_store or InMemoryIdempotencyStore()
