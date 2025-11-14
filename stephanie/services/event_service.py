# stephanie/services/event_service.py
from __future__ import annotations

import asyncio
import logging
import socket
import time
import traceback
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from stephanie.services.bus.idempotency import (JsonlIdempotencyStore,
                                                NatsKVIdempotencyStore)
from stephanie.services.service_protocol import Service

Handler = Callable[[dict], Awaitable[None]]
RpcHandler = Callable[[dict], Awaitable[dict]]

log = logging.getLogger(__name__)

class EventService(Service):
    """
    Uniform event-bus service with:
      - subject envelopes + idempotency
      - DLQ publishing
      - RPC helpers
      - health/metrics
    """

    SCHEMA_VERSION = "v1"

    def __init__(self, cfg: dict, memory, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger

        self._name = self.cfg.get("event_service_name", "event-service-v1")
        self.instance_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self.durable_prefix = self._name.replace(".", "_")
        self.dlq_enabled = bool(self.cfg.get("dlq_enabled", True))

        # lifecycle
        self._initialized = False
        self._starter: Optional[asyncio.Task] = None

        # metrics
        self._metrics = {
            "started_at": time.time(),
            "events_processed": 0,
            "events_failed": 0,
            "rpc_calls": 0,
            "rpc_failed": 0,
            "idempotent_skips": 0,
            "subscriptions": 0,
        }

        # idempotency (auto-pick best)
        if getattr(self.memory.bus, "is_nats", False) and getattr(self.memory.bus, "_js", None):
            self.idem = NatsKVIdempotencyStore(self.memory.bus._js)  # type: ignore
        else:
            self.idem = JsonlIdempotencyStore(
                f"./data/idempotency/{self._name}.jsonl"
            )

        self._subscriptions: list[tuple[str, Handler]] = []
        self._rpc_handlers: dict[str, RpcHandler] = {}

    # ============== Service Protocol ==============

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, **kwargs) -> None:
        """
        Start background bootstrap to connect bus, subscribe handlers, register RPC.
        Mirrors ReportingService.initialize pattern.
        """
        if self._initialized:
            return
        loop = asyncio.get_event_loop()
        self._starter = loop.create_task(self._async_start())
        self._initialized = True

    def health_check(self) -> Dict[str, Any]:
        up = time.time() - self._metrics["started_at"]
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                **self._metrics,
                "uptime_sec": round(up, 1),
                "bus_backend": "nats" if getattr(self.memory.bus, "is_nats", False) else "local",
                "routes_registered": len(self._subscriptions),
                "rpc_routes": len(self._rpc_handlers),
                "starter_running": bool(self._starter and not self._starter.done()),
            },
            "dependencies": {},
        }

    def shutdown(self) -> None:
        """
        Stop background bootstrap. (Bus may be shared; we do not close it here.)
        """
        if self._starter and not self._starter.done():
            self._starter.cancel()
        self._starter = None
        self._initialized = False
        # leave subscriptions as-is; upstream bus manages lifetimes

    # ============== Bootstrap (async) ==============

    async def _async_start(self) -> None:
        try:
            await self.memory.ensure_bus_connected()
            if not self.memory.bus:
                raise RuntimeError("EventService requires memory.bus")

            # subscribe events
            for subject, handler in self.event_routes():
                await self._subscribe(subject, handler)

            # register rpc (subject pattern handled by bus impl)
            self._rpc_handlers.update(self.rpc_routes())

            self._metrics["subscriptions"] = len(self._subscriptions)
            self.logger.info(
                f"[{self._name}] started with "
                f"{len(self._subscriptions)} event routes and "
                f"{len(self._rpc_handlers)} rpc routes"
            )
        except Exception as e:
            log.error(f"[{self._name}] init failed: {e}", exc_info=True)
            # mark unhealthy but don’t crash the process
            self._initialized = False

    # ============== Override points ==============

    def event_routes(self) -> list[tuple[str, Handler]]:
        """
        Return (subject, async handler) pairs.
        Use raw subject (e.g., 'events.something.happened').
        """
        return []

    def rpc_routes(self) -> dict[str, RpcHandler]:
        """
        Return name->handler mapping; bus will map to 'rpc.<name>'.
        """
        return {}

    # ============== Bus helpers (public) ==============

    async def publish(self, subject: str, payload: dict) -> None:
        await self.memory.bus.publish(subject, self._envelope(subject, payload))
        self.memory.bus_events.insert(subject, payload)

    async def request(
        self, rpc_name: str, payload: dict, timeout: float = 5.0
    ) -> Optional[dict]:
        self._metrics["rpc_calls"] += 1
        env = self._envelope(f"rpc.{rpc_name}", payload)
        try:
            return await self.memory.bus.request(rpc_name, env, timeout=timeout)
        except Exception:
            self._metrics["rpc_failed"] += 1
            raise

    async def add_route(self, subject: str, handler: Handler) -> None:
        """Dynamically subscribe a new subject with idempotency + DLQ."""
        await self._subscribe(subject, handler)

    async def remove_route(self, subject: str) -> None:
        """Best-effort unbind: keep our bookkeeping clean; underlying bus unsubscribe is optional."""
        self._subscriptions = [(s, h) for (s, h) in self._subscriptions if s != subject]

    # ============== Internals ==============

    async def _subscribe(self, subject: str, handler: Handler):
        async def _wrapped(enveloped: dict):
            try:
                # idempotency
                event_id = str(enveloped.get("event_id") or "")
                if event_id:
                    if await self.idem.seen(event_id):
                        self._metrics["idempotent_skips"] += 1
                        return
                    await self.idem.mark(event_id)

                payload = enveloped.get("payload", enveloped)
                await handler(payload)
                self._metrics["events_processed"] += 1
            except Exception as e:
                self._metrics["events_failed"] += 1
                self.logger.error(
                    f"[{self._name}] handler failed",
                    extra={
                        "subject": subject,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                if self.dlq_enabled:
                    await self._emit_dlq(subject, enveloped, error=str(e))

        await self.memory.bus.subscribe(subject, _wrapped)
        self._subscriptions.append((subject, handler))

    def _envelope(self, subject: str, payload: dict) -> dict:
        return {
            "event_id": uuid.uuid4().hex,
            "service": self._name,
            "instance": self.instance_id,
            "subject": subject,
            "ts": time.time(),
            "schema": self.SCHEMA_VERSION,
            "payload": payload,
        }

    async def _emit_dlq(self, subject: str, enveloped: dict, error: str):
        dlq_subject = f"dlq.{subject}"
        record = dict(enveloped)
        record["error"] = error
        await self.memory. OK (dlq_subject, record)
        self.memory.bus_events.insert(dlq_subject, record)

