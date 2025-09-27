# stephanie/services/bus/nats_bus.py
"""
NATS JetStream Bus Implementation – Production Ready

Features:
- Bounded concurrency for publish bursts
- Explicit PUBACK deadlines + retries (with jitter)
- Fire-and-forget fallback for telemetry
- Keepalive + auto-heal during long debug pauses
- Auto re-subscribe on reconnect (durables)
- Proper timedelta types for ack_wait
- Optional DLQ writer for final failures
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from nats.aio.client import Client as NATS
from nats.errors import ConnectionClosedError, NoServersError
from nats.errors import TimeoutError
from nats.errors import TimeoutError as NatsTimeoutError
from nats.js.api import ConsumerConfig, DeliverPolicy, RetentionPolicy
from nats.js.errors import NotFoundError

from .bus_protocol import BusProtocol
from .errors import (BusConnectionError, BusPublishError, BusRequestError,
                     BusSubscribeError)
from .idempotency import InMemoryIdempotencyStore, NatsKVIdempotencyStore


def _sanitize_durable(stream: str, subject: str) -> str:
    name = f"durable_{stream}_{subject}".replace(".", "_").replace(">", "all")
    # JetStream durable name limits are generous, but keep it sane:
    return name[:240]

_logger = logging.getLogger(__name__)

class NatsKnowledgeBus(BusProtocol):
    """
    Production-grade NATS JetStream bus with resilience + good DX while debugging.
    """

    def __init__(
        self,
        servers: List[str] = ["nats://localhost:4222"],
        stream: str = "stephanie",
        logger: Optional[logging.Logger] = None,
        *,
        timeout: float = 1.0,           # publish/request deadline (s)
        max_retries: int = 3,
        retry_base_delay: float = 0.2,  # backoff base
        max_in_flight: int = 256,       # bound concurrent publishes
        health_check_interval: float = 30.0,
        debug: bool = False,            # stickier settings for debug
        fire_and_forget_subjects: Optional[Set[str]] = None,  # subjects to bypass JS
        dlq_writer: Optional[Callable[[dict], None]] = None,  # optional JSONL writer
    ):
        self.servers = servers
        self.stream = stream

        if not debug:
            debug = self._is_debugger_attached()
        self.debug = debug

        self.timeout = 30.0 if debug else timeout
        self.max_retries = 50 if debug else max_retries
        self.retry_base_delay = retry_base_delay
        self.health_check_interval = health_check_interval
        self.debug = debug

        self._nc: Optional[NATS] = None
        self._js = None
        self._idem_store = None
        self._connected = False

        self._sem = asyncio.Semaphore(max_in_flight)
        self._health_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

        self._subscriptions: Dict[str, Dict[str, Any]] = {}  # subject -> intent (handler, durable_name)
        self._last_publish_time = 0.0
        self._publish_failures = 0
        self._dlq_writer = dlq_writer

        # Subjects that should use plain NATS (no PUBACK, no JS), useful for telemetry
        self._faf_subjects: Set[str] = fire_and_forget_subjects or set()

    # ---------- Connection management ----------

    async def connect(self) -> bool:
        if self._connected:
            return True

        # stop monitors if any (we'll restart on success)
        await self._stop_tasks()

        for attempt in range(self.max_retries + 1):
            try:
                if self._nc:
                    with contextlib.suppress(Exception):
                        await self._nc.close()
                    self._nc = None

                self._nc = NATS()
                await self._nc.connect(
                    servers=self.servers,
                    name="stephanie-nats-bus",
                    allow_reconnect=True,
                    reconnect_time_wait=2.0,
                    max_reconnect_attempts=-1 if self.debug else 10,
                    ping_interval=10,
                    max_outstanding_pings=5,
                    connect_timeout=3.0,
                    drain_timeout=2.0,
                    disconnected_cb=self._on_disconnected_cb,
                    reconnected_cb=self._on_reconnected_cb,
                    closed_cb=self._on_closed_cb,
                    error_cb=self._on_error_cb,
                )
                self._js = self._nc.jetstream()

                # Ensure stream exists (OK if it already exists)
                try:
                    await self._js.stream_info(self.stream)
                    _logger.info(f"JetStream stream '{self.stream}' present.")
                except NotFoundError:
                    await self._js.add_stream(
                        name=self.stream,
                        subjects=[f"{self.stream}.>"],
                        retention=RetentionPolicy.LIMITS,
                        max_msgs=-1,
                        max_bytes=1_073_741_824,  # 1 GB (tune for your box)
                        max_age=0,                # no TTL by default
                    )
                    _logger.info(f"Created JetStream stream '{self.stream}'.")

                # KV for idempotency
                self._idem_store = NatsKVIdempotencyStore(
                    self._js, bucket=f"{self.stream}_idem"
                )

                self._connected = True
                _logger.info(
                    "Connected to NATS JetStream (servers: %s)",
                    ", ".join(self.servers),
                )

                # start keepalive + health monitors
                self._start_keepalive()
                self._start_health_monitoring()
                # re-subscribe any stored intents (durables continue where they left off)
                await self._resubscribe_all()

                return True

            except (NoServersError, OSError, ConnectionClosedError, TimeoutError) as e:
                if attempt < self.max_retries:
                    wait_time = self._backoff(attempt)
                    _logger.warning(
                        "NATS connect attempt %d failed (%s). Retrying in %.2fs",
                        attempt + 1, type(e).__name__, wait_time
                    )
                    await asyncio.sleep(wait_time)
                    continue
                _logger.error("Failed to connect to NATS: %s", e, exc_info=True)
                return False
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = self._backoff(attempt)
                    _logger.warning(
                        "Unexpected error on connect (%s). Retrying in %.2fs",
                        type(e).__name__, wait_time, exc_info=True
                    )
                    await asyncio.sleep(wait_time)
                    continue
                _logger.error("Unexpected error connecting to NATS", exc_info=True)
                return False
        return False

    async def _on_reconnected(self):
        # Refresh JS context (defensive: cheap call)
        self._js = self._nc.jetstream()
        _logger.info("NATS reconnected; refreshing subscriptions.")
        await self._resubscribe_all()

    async def _resubscribe_all(self):
        # Rebind every stored subscription intent
        for subject, intent in list(self._subscriptions.items()):
            handler = intent["handler"]
            durable = intent["durable"]
            try:
                await self._do_subscribe(subject, handler, durable)
                _logger.debug("Re-subscribed to %s", subject)
            except Exception:
                _logger.exception("Failed to re-subscribe to %s", subject)

    async def _ensure_connected(self) -> None:
        """
        Lightweight health check that never raises CancelledError upstream.
        Uses NATS' own flush timeout instead of asyncio.wait_for, and
        treats cancellations as transient (debug-friendly).
        """
        if not self._nc or self._nc.is_closed or not self._connected:
            await self.connect()
            return
        try:
            # Use the client's own timeout handling; no wait_for wrapper.
            await self._nc.flush(timeout=1.0)
        except (NatsTimeoutError, asyncio.TimeoutError): 
            _logger.warning("NATS flush timeout; attempting reconnect")
            await self.connect()
        except asyncio.CancelledError:
            # Don't derail caller during debug/breakpoints. Mark unhealthy and return.
            _logger.debug("NATS flush cancelled; will reconnect on next op")
            self._connected = False
            return
        except Exception:
            # Any other error → reconnect path
            self._connected = False
            await self.connect()

    # ---------- Publish / Subscribe / Request ----------

    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """
        Publish with bounded concurrency, deadline, retries, and optional fallback.
        """
        start = time.time()

        await self._ensure_connected()

        envelope = {
            "event_id": f"{subject}-{uuid.uuid4().hex}",
            "timestamp": time.time(),
            "subject": subject,
            "payload": payload,
            "bus_backend": "nats",
        }

        # Plain NATS (no JS) for telemetry subjects if configured
        if subject in self._faf_subjects:
            data = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
            async with self._sem:
                await self._nc.publish(f"{self.stream}.{subject}", data)
                await self._nc.flush(timeout=min(self.timeout, 1.0))
                self._last_publish_time = time.time()
                self._publish_failures = 0
                _logger.debug("FAF published %s in %.3fs", subject, time.time() - start)
                return

        data = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        full_subject = f"{self.stream}.{subject}"
        _logger.info("BUS → %s : %s", full_subject, data[:200])
        async with self._sem:
            last_exc = None
            for attempt in range(self.max_retries + 1):
                try:
                    coro = self._js.publish(full_subject, data)
                    await asyncio.wait_for(coro, timeout=self.timeout)
                    self._last_publish_time = time.time()
                    self._publish_failures = 0
                    _logger.info(
                        "Published %s (id=%s) in %.3fs",
                        subject, envelope["event_id"], time.time() - start
                    )
                    return
                except (asyncio.TimeoutError, TimeoutError) as e:
                    _logger.exception("Error publishing to %s: %s", subject, e)
                    last_exc = e
                    if attempt < self.max_retries:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    # Final failure → DLQ + raise
                    self._write_dlq("publish_timeout", subject, envelope)
                    raise BusPublishError(f"Publish timeout for {subject}") from e
                except Exception as e:
                    _logger.exception("Error publishing to %s: %s", subject, e)
                    last_exc = e
                    if attempt < self.max_retries:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    self._write_dlq(f"publish_error:{type(e).__name__}", subject, envelope)
                    raise BusPublishError(f"Failed to publish to {subject}") from e

    async def request(self, subject: str, payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        await self._ensure_connected()
        request_timeout = min(timeout, self.timeout)

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._nc.request(f"{self.stream}.rpc.{subject}", data, timeout=request_timeout)
                return json.loads(resp.data.decode())
            except TimeoutError as e:
                last_exc = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                return None
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                raise BusRequestError(f"Request failed for {subject}") from e

    # ---------- Shutdown ----------

    async def close(self) -> None:
        await self._stop_tasks()
        if self._connected and self._nc:
            # best-effort unsubscribe
            for subject, meta in list(self._subscriptions.items()):
                sub = meta.get("sub")
                if sub:
                    with contextlib.suppress(Exception):
                        await sub.unsubscribe()
                    _logger.debug("Unsubscribed from %s", subject)
            self._subscriptions.clear()

            with contextlib.suppress(Exception):
                await self._nc.drain()
            with contextlib.suppress(Exception):
                await self._nc.close()
            self._connected = False
            _logger.info("NATS connection closed")

    # ---------- Helpers ----------

    def get_backend(self) -> str:
        return "nats"

    @property
    def idempotency_store(self) -> Any:
        return self._idem_store or InMemoryIdempotencyStore()


    def debug_connection_status(self) -> Dict[str, Any]:
        """Return detailed connection status for debugging"""
        return {
            "connected": self._connected,
            "last_publish": self._last_publish_time,
            "publish_failures": self._publish_failures,
            "connection_uptime": (
                time.time() - self._last_publish_time 
                if self._last_publish_time else 0
            ),
            "debug_mode": self.debug,
            "timeout": self.timeout,
            "reconnect_attempts": self.max_retries,
            "keepalive_interval": 10.0 if self.debug else 20.0,
            "health_check_interval": self.health_check_interval,
            "subscriptions": list(self._subscriptions.keys())
        }


    def _build_wrapped(self, subject: str, handler: Callable[[Dict[str, Any]], Any]):
        """
        Build the coroutine callback that JetStream requires.
        Handles idempotency and always acks safely.
        """
        async def wrapped(msg):
            try:
                envelope = json.loads(msg.data.decode())
                event_id = envelope.get("event_id")
                # Idempotency: skip duplicates
                if event_id and await self._idem_store.seen(event_id):
                    await msg.ack()
                    _logger.debug("Skipping duplicate event: %s", event_id)
                    return
                if event_id:
                    await self._idem_store.mark(event_id)

                # Invoke user handler with the payload only
                await handler(envelope["payload"])
            except Exception:
                # Avoid redelivery storms: log and still ack in finally.
                _logger.exception("Error handling event %s", subject)
            finally:
                with contextlib.suppress(Exception):
                    await msg.ack()
        return wrapped


    async def subscribe(self, subject: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        """
        Public subscribe: ensures connection, builds the wrapped cb,
        stores intent for auto re-subscribe, then binds the consumer.
        """
        await self._ensure_connected()

        durable_name = _sanitize_durable(self.stream, subject)
        wrapped_cb = self._build_wrapped(subject, handler)

        # Remember intent (for auto re-subscribe on reconnect)
        self._subscriptions[subject] = {
            "handler": handler,
            "durable": durable_name,
            # store the built callback so we reuse the exact same closure on reconnects
            "cb": wrapped_cb,
        }

        await self._do_subscribe(subject, durable_name, wrapped_cb)


    async def _do_subscribe(self, subject: str, durable_name: str, cb) -> None:
        """
        Internal binder: actually calls js.subscribe with a coroutine callback.
        """
        cfg = ConsumerConfig(
            deliver_policy=DeliverPolicy.ALL,
            ack_wait=timedelta(seconds=max(1, int(self.timeout * 5))),  # > publish timeout
            max_deliver=self.max_retries + 1,
        )
        full_subject = f"{self.stream}.{subject}"

        # IMPORTANT: cb MUST be a coroutine function, not a lambda
        sub = await self._js.subscribe(
            full_subject,
            durable=durable_name,
            cb=cb,
            config=cfg,
        )

        # Keep the subscription object for graceful unsubscribe on close
        self._subscriptions[subject]["sub"] = sub
        _logger.info("Subscribed to %s (durable=%s)", subject, durable_name)


    # If you have a re-subscribe path, make sure it reuses the stored 'cb':
    async def _resubscribe_all(self):
        for subject, meta in list(self._subscriptions.items()):
            handler = meta["handler"]
            durable = meta["durable"]
            cb = meta.get("cb") or self._build_wrapped(subject, handler)
            try:
                await self._do_subscribe(subject, durable, cb)
                _logger.debug("Re-subscribed to %s", subject)
            except Exception:
                _logger.exception("Failed to re-subscribe to %s", subject)


    # at class scope
    async def _on_disconnected_cb(self, *args, **kwargs):
        _logger.warning("NATS disconnected")

    async def _on_reconnected_cb(self, *args, **kwargs):
        # Refresh JS and re-subscribe after reconnect
        try:
            self._js = self._nc.jetstream()
        except Exception:
            _logger.exception("Failed to refresh JetStream context on reconnect")
        await self._resubscribe_all()
        _logger.info("NATS reconnected; subscriptions refreshed")

    async def _on_closed_cb(self, *args, **kwargs):
        _logger.warning("NATS connection closed")

    async def _on_error_cb(self, e):
        # Note: signature is (Exception)
        _logger.error("NATS error: %r", e)


    def _backoff(self, attempt: int) -> float:
        # jittered exponential backoff
        base = self.retry_base_delay * (2 ** attempt)
        return base * (1 + 0.2 * random.random())

    def _write_dlq(self, error: str, subject: str, envelope: dict) -> None:
        if not self._dlq_writer:
            return
        try:
            self._dlq_writer(
                {
                    "ts": time.time(),
                    "error": error,
                    "subject": subject,
                    "envelope": envelope,
                }
            )
        except Exception:
            _logger.exception("DLQ writer failed for subject %s", subject)

    def _start_keepalive(self):
        if self._keepalive_task and not self._keepalive_task.done():
            return

        async def _loop():
            try:
                while True:
                    try:
                        # not connected – give connect() a chance on next op
                        if not self._connected or not self._nc or self._nc.is_closed:
                            await asyncio.sleep(3.0)
                            continue

                        # Use NATS' own timeout; do NOT wrap with wait_for
                        await self._nc.flush(timeout=1.0)
                        await asyncio.sleep(10.0 if self.debug else 20.0)

                    except (NatsTimeoutError, asyncio.TimeoutError):
                        # soft mark as unhealthy; next op will reconnect
                        self._connected = False
                        await asyncio.sleep(2.0)

                    except asyncio.CancelledError:
                        # graceful task shutdown
                        break

                    except Exception:
                        # any other error → mark unhealthy and back off a bit
                        self._connected = False
                        await asyncio.sleep(2.0)
            finally:
                # optional: any cleanup here
                pass

        self._keepalive_task = asyncio.create_task(_loop())
        _logger.debug("Keepalive started")

    def _start_health_monitoring(self):
        if self._health_task and not self._health_task.done():
            return

        async def _health():
            try:
                while True:
                    try:
                        await asyncio.sleep(self.health_check_interval)

                        if not self._connected or not self._nc or self._nc.is_closed:
                            # Let normal ops trigger reconnect; don't spam here
                            continue

                        # Ping only; do NOT wrap with wait_for
                        await self._nc.flush(timeout=1.0)

                    except (NatsTimeoutError, asyncio.TimeoutError):
                        # mark unhealthy; next bus op will reconnect
                        self._connected = False

                    except asyncio.CancelledError:
                        break

                    except Exception:
                        # conservative: mark unhealthy but don't crash task
                        self._connected = False
            finally:
                pass

        self._health_task = asyncio.create_task(_health())
        _logger.debug("Health monitor started")

    async def _stop_tasks(self):
        for t in (self._keepalive_task, self._health_task):
            if t:
                t.cancel()
                with contextlib.suppress(Exception):
                    await t
        self._keepalive_task = None
        self._health_task = None

    async def _safe_flush(self, timeout: float = 1.0) -> bool:
        """
        Try to flush with NATS' own timeout; never raises CancelledError to callers.
        Returns True if OK, False if it timed out or errored.
        """
        if not self._nc or self._nc.is_closed:
            return False
        try:
            await self._nc.flush(timeout=timeout)
            return True
        except (NatsTimeoutError, asyncio.TimeoutError):
            return False
        except asyncio.CancelledError:
            return False
        except Exception:
            return False


    @staticmethod
    def _is_debugger_attached() -> bool:
        """Detect if a debugger is attached (PyCharm, VSCode, etc.)"""
        try:
            # Common debugger detection methods
            return (
                hasattr(sys, 'gettrace') and sys.gettrace() is not None or
                'pydevd' in sys.modules or
                'pdb' in sys.modules or
                os.getenv('DEBUG', '0') == '1'
            )
        except:
            return False

def jsonl_dlq_writer_factory(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    def write(obj: dict):
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return write

