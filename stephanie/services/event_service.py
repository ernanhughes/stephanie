# stephanie/services/event_service.py
from __future__ import annotations

import socket
import time
import traceback
import uuid
from typing import Awaitable, Callable, Optional

from stephanie.services.bus.hybrid_bus import HybridKnowledgeBus
from stephanie.services.bus.idempotency import (IdempotencyStore,
                                                JsonlIdempotencyStore,
                                                NatsKVIdempotencyStore)

Handler = Callable[[dict], Awaitable[None]]
RpcHandler = Callable[[dict], Awaitable[dict]]

class EventService:
    """
    Base class for all services that use the event bus.
    - Uniform subject naming, envelopes, idempotency, DLQ, metrics
    - Works with NATS JetStream if available, else local in-process bus
    """

    SCHEMA_VERSION = "v1"

    def __init__(
        self,
        service_name: str,
        bus: HybridKnowledgeBus,
        logger,
        durable_prefix: str | None = None,
        idempotency: IdempotencyStore | None = None,
        dlq_enabled: bool = True,
    ):
        self.service_name = service_name
        self.bus = bus
        self.logger = logger
        self.instance_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self.durable_prefix = durable_prefix or service_name.replace(".", "_")
        self.dlq_enabled = dlq_enabled

        # metrics
        self._metrics = {
            "started_at": time.time(),
            "events_processed": 0,
            "events_failed": 0,
            "rpc_calls": 0,
            "rpc_failed": 0,
            "idempotent_skips": 0,
        }

        # idempotency (auto-pick best)
        if idempotency:
            self.idem = idempotency
        else:
            if getattr(self.bus, "is_nats", False) and getattr(self.bus, "_js", None):
                self.idem = NatsKVIdempotencyStore(self.bus._js)  # type: ignore
            else:
                # persistent JSONL > memory (for across-runs continuity)
                self.idem = JsonlIdempotencyStore(f"./data/idempotency/{self.service_name}.jsonl")

        self._subscriptions: list[tuple[str, Handler]] = []
        self._rpc_handlers: dict[str, RpcHandler] = {}

    # ---------- override points ----------

    def event_routes(self) -> list[tuple[str, Handler]]:
        """
        Return (subject, async handler) pairs.
        Subject is WITHOUT the 'events.' prefix (base adds it).
        """
        return []

    def rpc_routes(self) -> dict[str, RpcHandler]:
        """
        Return name->handler mapping for RPC (subject is 'rpc.<name>').
        """
        return {}

    # ---------- lifecycle ----------

    async def start(self):
        await self.bus.connect()
        # subscribe events
        for subject, handler in self.event_routes():
            await self._subscribe(subject, handler)
        # register rpc
        self._rpc_handlers.update(self.rpc_routes())
        if not self._subscriptions and not self._rpc_handlers:
            self.logger.info(f"[{self.service_name}] started (no routes)")
        else:
            self.logger.info(f"[{self.service_name}] started with {len(self._subscriptions)} event routes and {len(self._rpc_handlers)} rpc routes")

    async def stop(self):
        self.logger.info(f"[{self.service_name}] stopping")
        # bus may be shared; don't close it here if managed elsewhere

    # ---------- bus helpers ----------

    async def publish(self, subject: str, payload: dict) -> None:
        await self.bus.publish(subject, self._envelope(subject, payload))

    async def request(self, rpc_name: str, payload: dict, timeout: float = 5.0) -> Optional[dict]:
        self._metrics["rpc_calls"] += 1
        env = self._envelope(f"rpc.{rpc_name}", payload)
        res = await self.bus.request(rpc_name, env, timeout=timeout)
        return res

    # ---------- internals ----------

    async def _subscribe(self, subject: str, handler: Handler):
        durable = f"{self.durable_prefix}_{subject.replace('.', '_')}"
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
                    f"[{self.service_name}] handler failed",
                    extra={"subject": subject, "error": str(e), "traceback": traceback.format_exc()}
                )
                if self.dlq_enabled:
                    await self._emit_dlq(subject, enveloped, error=str(e))

        await self.bus.subscribe(subject, _wrapped)
        self._subscriptions.append((subject, handler))

    def _envelope(self, subject: str, payload: dict) -> dict:
        return {
            "event_id": uuid.uuid4().hex,
            "service": self.service_name,
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
        await self.bus.publish(dlq_subject, record)

    # ---------- diagnostics ----------

    def health(self) -> dict:
        up = time.time() - self._metrics["started_at"]
        return {
            "service": self.service_name,
            "instance": self.instance_id,
            "uptime_sec": round(up, 1),
            "metrics": dict(self._metrics),
            "bus": "nats" if getattr(self.bus, "is_nats", False) else "local",
        }
