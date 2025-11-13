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
- Truthy connect() only after JetStream is fully ready (stream ensured + health ping)
- Quietly handles asyncio.CancelledError during shutdown/timeout (returns False)
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
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from nats import errors as nats_errors
from nats.aio.client import Client as NATS
from nats.errors import TimeoutError as NatsTimeoutError
from nats.js.api import (
    ConsumerConfig,
    DeliverPolicy,
    RetentionPolicy,
    StreamConfig,
)

from .bus_protocol import BusProtocol
from .errors import BusRequestError
from .idempotency import InMemoryIdempotencyStore

log = logging.getLogger(__name__)


def _sanitize_durable(stream: str, subject: str) -> str:
    name = f"durable_{stream}_{subject}".replace(".", "_").replace(">", "all")
    return name[:240]


def _as_timedelta(v: Optional[Union[int, float, timedelta]], default_seconds: float) -> timedelta:
    if v is None:
        return timedelta(seconds=int(max(1, default_seconds)))
    if isinstance(v, timedelta):
        return v
    try:
        return timedelta(seconds=float(v))
    except Exception:
        return timedelta(seconds=int(max(1, default_seconds)))


def _ack_wait_to_ns(v: Optional[Union[int, float, timedelta]], default_seconds: float) -> int:
    """
    Accept timedelta | float | int | None and return an int nanoseconds value
    compatible with current nats-py ConsumerConfig. Defaults to seconds.
    """
    if v is None:
        return int(default_seconds * 1e9)
    if isinstance(v, timedelta):
        return int(v.total_seconds() * 1e9)
    if isinstance(v, (int, float)):
        return int(float(v) * 1e9)  # treat as seconds
    # Last resort: try to coerce strings like "5"
    return int(float(v) * 1e9)


class NatsKnowledgeBus(BusProtocol):
    """
    Production-grade NATS JetStream bus with resilience + good DX while debugging.
    All logs use structured `logger.<level>(event, data)` (compatible with JSONLogger).
    """

    def __init__(
        self,
        servers: List[str] = ["nats://localhost:4222"],
        stream: str = "stephanie",
        logger: Optional[logging.Logger] = None,
        *,
        timeout: float = 1.0,            # publish/request deadline (s)
        max_retries: int = 3,
        retry_base_delay: float = 0.2,   # backoff base
        max_in_flight: int = 256,        # bound concurrent publishes
        health_check_interval: float = 30.0,
        debug: bool = False,             # stickier settings for debug
        fire_and_forget_subjects: Optional[Set[str]] = None,  # subjects to bypass JS
        dlq_writer: Optional[Callable[[dict], None]] = None,  # optional JSONL writer
    ):
        self.servers = servers
        self.stream = stream
        self.logger = logger

        if not debug:
            debug = self._is_debugger_attached()
        self.debug = debug

        self.timeout = 30.0 if debug else timeout
        self.max_retries = 50 if debug else max_retries
        self.retry_base_delay = retry_base_delay
        self.health_check_interval = health_check_interval

        self._nc: Optional[NATS] = None
        self._js = None
        self._idem_store = InMemoryIdempotencyStore()
        self._connected = False

        self._sem = asyncio.Semaphore(max_in_flight)
        self._health_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

        # subject -> {handler, durable, cb, sub, queue, ack_wait, max_deliver, deliver_policy}
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._last_publish_time = 0.0
        self._publish_failures = 0
        self._dlq_writer = dlq_writer

        # Subjects that should use plain NATS (no PUBACK, no JS), useful for telemetry
        self._faf_subjects: Set[str] = fire_and_forget_subjects or set()
        self._tasks: Set[asyncio.Task] = set()

        # Connection guards
        self._conn_lock = asyncio.Lock()

        # Stream subjects we ensure; default to "<stream>.>"
        self._stream_subjects: List[str] = [f"{self.stream}.>"]
        self._stopping: bool = False

    # ---------- Connection management ----------

    async def connect(self, force: bool = False) -> bool:
        """
        Connects to NATS with intelligent backoff and cancellation handling.
        Returns True only after full readiness (connection + stream + health check).
        """
        # Quick check for existing valid connection
        if self._connected and self._nc and not self._nc.is_closed and not force:
            return True

        async with self._conn_lock:
            # Double-check pattern after acquiring lock
            if self._connected and self._nc and not self._nc.is_closed and not force:
                return True

            # Initialize backoff state if needed
            if not hasattr(self, "_backoff_state"):
                self._backoff_state = {
                    "attempts": 0,
                    "last_attempt": 0.0,
                    "base_delay": 0.1  # Start with 100ms
                }

            # Calculate appropriate backoff delay
            current_time = time.time()
            if self._backoff_state["attempts"] > 0:
                delay = min(5.0, self._backoff_state["base_delay"] * (2 ** (self._backoff_state["attempts"] - 1)))
                # Add jitter to prevent synchronized retries
                delay = delay * (0.8 + 0.4 * random.random())
                
                # Only delay if it's been less than the delay period since last attempt
                if current_time - self._backoff_state["last_attempt"] < delay:
                    log.debug("Applying backoff delay of %.2f seconds before connection attempt", delay)
                    await asyncio.sleep(delay)
            
            # Record attempt time
            self._backoff_state["last_attempt"] = time.time()

            # If forcing, drain old connection
            if self._nc and not self._nc.is_closed and force:
                with contextlib.suppress(Exception):
                    await self._nc.drain()

            # Establish a fresh connection and only commit on full readiness
            import nats

            async def _err_cb(e):
                log.debug(f"NATSLowLevelError error: {repr(e)}")

            async def _disc_cb():
                log.debug("NATSDisconnected")
                # Reset backoff on disconnection (we'll need to reconnect)
                self._backoff_state["attempts"] = 0

            async def _reconn_cb():
                log.debug("NATSReconnected")
                # Reset backoff on successful reconnection
                self._backoff_state["attempts"] = 0
                await self._on_reconnected()

            async def _closed_cb():
                log.debug("NATSClosed")
                # Reset backoff when connection is properly closed
                self._backoff_state["attempts"] = 0

            nc = None
            try:
                nc = await nats.connect(
                    servers=self.servers,
                    name=self.stream,
                    allow_reconnect=True,
                    max_reconnect_attempts=-1,
                    reconnect_time_wait=0.5,
                    ping_interval=5,
                    max_outstanding_pings=2,
                    error_cb=_err_cb,
                    disconnected_cb=_disc_cb,
                    reconnected_cb=_reconn_cb,
                    closed_cb=_closed_cb,
                )
                js = nc.jetstream()

                # Ensure stream exists; create if missing
                try:
                    await js.stream_info(self.stream)
                except nats.js.errors.NotFoundError:
                    cfg = StreamConfig(
                        name=self.stream,
                        subjects=self._stream_subjects,
                        retention=RetentionPolicy.LIMITS,
                    )
                    await js.add_stream(cfg)
                except Exception as e:
                    log.error("Failed to verify/create stream: %r", e)
                    with contextlib.suppress(Exception):
                        await nc.close()
                    # Increment backoff counter for stream issues
                    self._backoff_state["attempts"] += 1
                    return False

                # Health publish to confirm JS perms
                health_subject = f"{self.stream}.health"
                try:
                    await js.publish(health_subject, b"ping")
                except Exception as e:
                    log.error("NATSHealthPublishFailed subject: %s, error: %r", health_subject, e)
                    with contextlib.suppress(Exception):
                        await nc.close()
                    # Increment backoff counter for health check failures
                    self._backoff_state["attempts"] += 1
                    return False

                # Connection successful - reset backoff counter
                self._backoff_state["attempts"] = 0
                
                # Commit only now
                self._nc = nc
                self._js = js
                self._connected = True
                log.info("NATSConnected servers: %s, stream: %s", self.servers, self.stream)

                # Start monitors (idempotent)
                self._start_keepalive()
                self._start_health_monitoring()
                return True

            except Exception as e:
                log.warning("NATS connection attempt failed: %r", e)
                # Increment backoff counter for all other exceptions
                self._backoff_state["attempts"] += 1
                
                with contextlib.suppress(Exception):
                    if nc and not nc.is_closed:
                        await nc.close()
                return False
        
    async def _on_reconnected(self):
        # Refresh JS context and re-subscribe
        try:
            if self._nc:
                self._js = self._nc.jetstream()
        except Exception as e:
            log.error("NATSRefreshJSOnReconnectFailed error: %r", e)
        await self._resubscribe_all()
        log.debug("NATSReconnectedSubscriptionsRefreshed")

    async def _ensure_connected(self) -> None:
        if self._connected and self._nc and not self._nc.is_closed:
            return
        ok = await self.connect()
        if not ok:
            log.error("NATSEnsureConnectFailed")

    # ---------- Publish / Subscribe / Request ----------

    def _maybe_prefix(self, subject: str) -> str:
        """Ensure subjects for JS are under '<stream>.' without double-prefixing."""
        prefix = f"{self.stream}."
        return subject if subject.startswith(prefix) else f"{self.stream}.{subject}"

    async def _try_nc_publish(self, subject: str, data: bytes, headers: Optional[Dict[str, str]]) -> None:
        # Core NATS publish with optional headers; fallback for older clients
        if headers:
            try:
                await self._nc.publish(subject, data, headers=headers)  # type: ignore[call-arg]
                return
            except TypeError:
                log.debug("NATSNoHeaderSupportCore", {"subject": subject})
        await self._nc.publish(subject, data)

    async def _try_js_publish(self, subject: str, data: bytes, headers: Optional[Dict[str, str]]) -> None:
        # JetStream publish with optional headers; fallback for older clients
        if headers:
            try:
                await self._js.publish(subject, data, headers=headers)  # type: ignore[call-arg]
                return
            except TypeError:
                log.debug("NATSNoHeaderSupportJS", {"subject": subject})
        await self._js.publish(subject, data)

    async def publish(
        self,
        subject: str,
        payload: dict,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Publish a JSON envelope. If `subject` is in fire-and-forget, use core NATS.
        Otherwise use JetStream on '<stream>.<subject>'. Pass headers when supported.
        """
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await self._ensure_connected()
        if not (self._nc and not self._nc.is_closed):
            self._publish_failures += 1
            log.error("NATSPublishSkippedNotConnected subject: %s", subject)
            return

        async with self._sem:
            try:
                if subject in self._faf_subjects:
                    await self._try_nc_publish(subject, data, headers)
                else:
                    full_subject = self._maybe_prefix(subject)
                    await self._try_js_publish(full_subject, data, headers)
                self._last_publish_time = time.time()

            except (
                nats_errors.TimeoutError,
                nats_errors.FlushTimeoutError,
                nats_errors.ConnectionClosedError,
                nats_errors.NoServersError,
                ConnectionResetError,
            ) as e:
                log.error("NATSPublishErrorRetrying subject: %s, error: %r", subject, e)
                ok = await self.connect(force=True)
                if not ok:
                    self._publish_failures += 1
                    self._write_dlq("publish_connect_failed", subject, payload)
                    return
                # Retry once after reconnect
                if subject in self._faf_subjects:
                    await self._try_nc_publish(subject, data, headers)
                else:
                    await self._try_js_publish(self._maybe_prefix(subject), data, headers)
                self._last_publish_time = time.time()

    async def publish_raw(
        self,
        subject: str,
        body: bytes,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Publish raw bytes. Honors fire-and-forget and headers (when supported).
        """
        await self._ensure_connected()
        if not (self._nc and not self._nc.is_closed):
            self._publish_failures += 1
            log.error("NATSPublishRawSkippedNotConnected subject: %s", subject)
            return

        async with self._sem:
            try:
                if subject in self._faf_subjects:
                    await self._try_nc_publish(subject, body, headers)
                else:
                    await self._try_js_publish(self._maybe_prefix(subject), body, headers)
                self._last_publish_time = time.time()

            except (
                nats_errors.TimeoutError,
                nats_errors.FlushTimeoutError,
                nats_errors.ConnectionClosedError,
                nats_errors.NoServersError,
                ConnectionResetError,
            ) as e:
                log.error("NATSPublishRawErrorRetrying subject: %s, error: %r", subject, e)
                ok = await self.connect(force=True)
                if not ok:
                    self._publish_failures += 1
                    self._write_dlq("publish_raw_connect_failed", subject, {"len": len(body)})
                    return
                # Retry once after reconnect
                if subject in self._faf_subjects:
                    await self._try_nc_publish(subject, body, headers)
                else:
                    await self._try_js_publish(self._maybe_prefix(subject), body, headers)
                self._last_publish_time = time.time()

    async def request(
        self, subject: str, payload: Dict[str, Any], timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Request/Reply over a namespaced RPC subject: "<stream>.rpc.<subject>"
        Retries with jittered backoff.
        """
        await self._ensure_connected()
        if not (self._nc and not self._nc.is_closed):
            log.error("NATSRequestSkippedNotConnected subject: %s", subject)
            return None

        request_timeout = min(timeout, self.timeout)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        rpc_subject = f"{self.stream}.rpc.{subject}"

        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._nc.request(rpc_subject, data, timeout=request_timeout)
                return json.loads(resp.data.decode())
            except NatsTimeoutError:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                log.error("NATSRequestTimeout subject: %s, attempts: %d", subject, attempt + 1)
                return None
            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                log.error("NATSRequestFailed subject: %s, error: %r", subject, e)
                raise BusRequestError(f"Request failed for {subject}") from e

    async def subscribe(
        self,
        subject: str,
        handler: Callable[[Dict[str, Any]], Any],
        *,
        # alias set — any of these may be used by callers
        queue: Optional[str] = None,
        queue_group: Optional[str] = None,
        deliver_group: Optional[str] = None,
        durable: Optional[str] = None,
        ack_wait: Optional[Union[int, float, timedelta]] = None,
        max_deliver: Optional[int] = None,
        deliver_policy: Optional[DeliverPolicy] = None,
    ) -> None:
        await self._ensure_connected()
        if not (self._js and self._nc and not self._nc.is_closed):
            log.error("NATSSubscribeSkippedNotConnected subject: %s", subject)
            return

        qgroup = queue or deliver_group or queue_group
        durable_name = durable or _sanitize_durable(self.stream, subject)
        wrapped_cb = self._build_wrapped(subject, handler)

        # remember intent for reconnects
        self._subscriptions[subject] = {
            "handler": handler,
            "durable": durable_name,
            "queue": qgroup,
            "ack_wait": ack_wait,
            "max_deliver": max_deliver,
            "deliver_policy": deliver_policy,
            "cb": wrapped_cb,
        }

        await self._do_subscribe(
            subject,
            durable_name,
            wrapped_cb,
            queue=qgroup,
            ack_wait=ack_wait,
            max_deliver=max_deliver,
            deliver_policy=deliver_policy,
        )

    async def _do_subscribe(
        self,
        subject: str,
        durable_name: str,
        cb,
        *,
        queue: Optional[str] = None,
        ack_wait: Optional[Union[int, float, timedelta]] = None,
        max_deliver: Optional[int] = None,
        deliver_policy: Optional[DeliverPolicy] = None,
    ) -> None:
        cfg = ConsumerConfig(
            deliver_policy=deliver_policy or DeliverPolicy.ALL,
            ack_wait=_ack_wait_to_ns(ack_wait, default_seconds=(self.timeout * 5)),
            max_deliver=(self.max_retries + 1) if max_deliver is None else int(max_deliver),
        )
        full_subject = f"{self.stream}.{subject}"

        try:
            sub = await self._js.subscribe(            # type: ignore[call-arg]
                full_subject,
                durable=durable_name,                  # 'durable' is broadly accepted
                cb=cb,
                config=cfg,
                queue=queue,                           # queue group if provided
            )
        except TypeError:
            # very old libs: drop queue kw
            sub = await self._js.subscribe(
                full_subject,
                durable=durable_name,
                cb=cb,
                config=cfg,
            )

        self._subscriptions[subject]["sub"] = sub
        self.logger.info("NATSSubscribed", {"subject": subject, "durable": durable_name, "queue": queue})

    async def _resubscribe_all(self):
        for subject, meta in list(self._subscriptions.items()):
            handler = meta["handler"]
            durable = meta["durable"]
            cb = meta.get("cb") or self._build_wrapped(subject, handler)
            try:
                await self._do_subscribe(
                    subject,
                    durable,
                    cb,
                    queue=meta.get("queue"),
                    ack_wait=meta.get("ack_wait"),
                    max_deliver=meta.get("max_deliver"),
                    deliver_policy=meta.get("deliver_policy"),
                )
                log.debug("NATSResubscribed subject: %s", subject)
            except Exception as e:
                log.error("NATSResubscribeFailed subject: %s, error: %r", subject, e)

    # ---------- Shutdown ----------

    async def stop(self) -> None:
        """Cooperative shutdown: stop loops first, then drain/close NATS."""
        self._stopping = True
        await self._stop_tasks()  # cancels keepalive/health tasks
        if self._nc:
            try:
                for subject, meta in list(self._subscriptions.items()):
                    sub = meta.get("sub")
                    if sub:
                        with contextlib.suppress(Exception):
                            await sub.unsubscribe()
                self._subscriptions.clear()
                with contextlib.suppress(Exception):
                    await self._nc.drain()
            finally:
                with contextlib.suppress(Exception):
                    await self._nc.close()
                self._connected = False
                self._nc = None
                self._js = None

    async def close(self) -> None:
        self._stopping = True
        await self._stop_tasks()
        if self._nc:
            try:
                for subject, meta in list(self._subscriptions.items()):
                    sub = meta.get("sub")
                    if sub:
                        with contextlib.suppress(Exception):
                            await sub.unsubscribe()
                self._subscriptions.clear()
                await self._nc.drain()
            except Exception:
                with contextlib.suppress(Exception):
                    await self._nc.close()
            finally:
                self._connected = False
                self._nc = None
                self._js = None
                log.debug("NATSClosedCleanly")

    # ---------- Helpers & Health ----------

    def get_backend(self) -> str:
        return "nats"

    @property
    def idempotency_store(self) -> Any:
        return self._idem_store

    def debug_connection_status(self) -> Dict[str, Any]:
        """Return detailed connection status for debugging"""
        return {
            "connected": self._connected,
            "last_publish": self._last_publish_time,
            "publish_failures": self._publish_failures,
            "connection_uptime": (time.time() - self._last_publish_time) if self._last_publish_time else 0,
            "debug_mode": self.debug,
            "timeout": self.timeout,
            "reconnect_attempts": self.max_retries,
            "keepalive_interval": 10.0 if self.debug else 20.0,
            "health_check_interval": self.health_check_interval,
            "subscriptions": list(self._subscriptions.keys()),
        }

    def _build_wrapped(self, subject: str, handler: Callable[[Dict[str, Any]], Any]):
        async def wrapped(msg):
            try:
                # Try JSON first; if it fails, deliver raw bytes to handler.
                try:
                    envelope = json.loads(msg.data.decode())
                except Exception:
                    await handler(msg.data)
                    await msg.ack()
                    return

                # Idempotency (best-effort if event_id exists)
                event_id = envelope.get("event_id")
                if event_id and await self.idempotency_store.seen(event_id):
                    await msg.ack()
                    log.debug("NATSDuplicateEventSkipped", {"event_id": event_id})
                    return
                if event_id:
                    await self.idempotency_store.mark(event_id)

                payload = envelope.get("payload", envelope)
                await handler(payload)

            except Exception as e:
                self.logger.error("NATSHandlerError", {"subject": subject, "error": repr(e)})
            finally:
                with contextlib.suppress(Exception):
                    await msg.ack()
        return wrapped

    async def _on_disconnected_cb(self, *args, **kwargs):
        log.error("NATSDisconnected")

    async def _on_reconnected_cb(self, *args, **kwargs):
        try:
            self._js = self._nc.jetstream()
        except Exception as e:
            log.error(f"NATSRefreshJSOnReconnectFailed error: {repr(e)}")
        await self._resubscribe_all()
        log.debug("NATSReconnectedSubscriptionsRefreshed")

    async def _on_closed_cb(self, *args, **kwargs):
        log.error("NATSClosed")

    async def _on_error_cb(self, e):
        log.error(f"NATSLowLevelError error: {repr(e)}")

    # ---------- Readiness / Ensure APIs ----------

    @property
    def is_connected(self) -> bool:
        return bool(self._connected and self._nc and not self._nc.is_closed)

    async def wait_ready(self, timeout: float = 5.0) -> bool:
        """
        Consider NATS 'ready' when:
          - TCP connected and not closed
          - JetStream available
          - Stream exists (auto-created if missing)
          - A tiny publish on '<stream>.health' succeeds
        """
        try:
            await self._ensure_connected()
            if not self.is_connected or not self._js:
                return False

            # Ensure stream with current subject set
            try:
                await asyncio.wait_for(self._js.stream_info(self.stream), timeout=timeout)
            except Exception:
                cfg = StreamConfig(name=self.stream, subjects=self._stream_subjects, retention=RetentionPolicy.LIMITS)
                await asyncio.wait_for(self._js.add_stream(cfg), timeout=timeout)

            # Health ping
            try:
                await asyncio.wait_for(self._js.publish(f"{self.stream}.health", b"ping"), timeout=timeout)
                return True
            except Exception as e:
                log.error("NATS wait_ready health publish failed: %r", e)
                return False
        except Exception:
            return False

    async def ensure_stream(self, stream: str, subjects: List[str]) -> bool:
        """
        Ensure a JetStream stream with the provided subject list exists.
        Idempotent: will create if missing, update subjects if needed.
        """
        await self._ensure_connected()
        if not self.is_connected or not self._js:
            return False

        # cache locally for future connects
        self._stream_subjects = subjects or [f"{stream}.>"]

        try:
            si = await self._js.stream_info(stream)
            # If subjects differ, try to update (best-effort; some servers disallow)
            have = set(getattr(si.config, "subjects", []) or [])
            want = set(self._stream_subjects)
            if have != want:
                try:
                    cfg = StreamConfig(name=stream, subjects=list(want), retention=RetentionPolicy.LIMITS)
                    await self._js.update_stream(cfg)  # may not exist on older libs
                except Exception:
                    # fallback: delete and recreate (danger: purges data)
                    with contextlib.suppress(Exception):
                        await self._js.delete_stream(stream)
                    cfg = StreamConfig(name=stream, subjects=list(want), retention=RetentionPolicy.LIMITS)
                    await self._js.add_stream(cfg)
            return True
        except Exception:
            # create new
            try:
                cfg = StreamConfig(name=stream, subjects=self._stream_subjects, retention=RetentionPolicy.LIMITS)
                await self._js.add_stream(cfg)
                return True
            except Exception as e:
                log.error("NATS ensure_stream failed: %r", e)
                return False

    async def ensure_consumer(
        self,
        stream: str,
        subject: str,
        durable: str,
        *,
        ack_wait: Optional[int] = None,
        max_deliver: Optional[int] = None,
        deliver_group: Optional[str] = None,
        deliver_policy: Optional[DeliverPolicy] = None,
    ) -> bool:
        await self._ensure_connected()
        if not self.is_connected or not self._js:
            return False

        cfg = ConsumerConfig(
            # Some nats-py versions accept 'durable_name', others 'durable'.
            # Prefer durable_name, fall back below if needed.
            durable_name=durable,
            filter_subject=f"{stream}.{subject}",
            deliver_policy=deliver_policy or DeliverPolicy.ALL,
            ack_wait=_ack_wait_to_ns(ack_wait, default_seconds=(self.timeout * 5)),
            max_deliver=(self.max_retries + 1) if max_deliver is None else int(max_deliver),
        )

        try:
            try:
                await self._js.consumer_info(stream, durable)
                with contextlib.suppress(Exception):
                    await self._js.update_consumer(stream, cfg)  # not in older libs
                return True
            except Exception:
                # Fallback for libs that want 'durable' kw instead of 'durable_name'
                try:
                    return bool(await self._js.add_consumer(stream, cfg))
                except TypeError:
                    cfg2 = ConsumerConfig(
                        durable=durable,
                        filter_subject=f"{stream}.{subject}",
                        deliver_policy=deliver_policy or DeliverPolicy.ALL,
                        ack_wait=_ack_wait_to_ns(ack_wait, default_seconds=(self.timeout * 5)),
                        max_deliver=(self.max_retries + 1) if max_deliver is None else int(max_deliver),
                    )
                    await self._js.add_consumer(stream, cfg2)
                    return True
        except Exception as e:
            log.error("NATS ensure_consumer failed: %r", e)
            return False

    # ---------- Flush / Drain Helpers ----------

    async def flush(self, timeout: float = 1.0) -> bool:
        """Wait for all pending messages and PUBACKs to be processed."""
        return await self._safe_flush(timeout=timeout)

    async def drain_subject(self, subject: str) -> bool:
        """
        Drain (purge or flush) pending JetStream messages for a subject.
        Works across all nats-py versions. Purges entire stream when filter unsupported.
        """
        await self._ensure_connected()
        full_subject = f"{self.stream}.{subject}"
        try:
            if self._js:
                try:
                    # Purge entire stream (subject filters not universally supported)
                    await self._js.purge_stream(self.stream)
                    log.debug("[NATSBus] Stream '%s' purged (no subject filter).", self.stream)
                except TypeError:
                    log.warning("[NATSBus] purge_stream() unsupported; flushing %s.", full_subject)
                    await self._safe_flush(timeout=2.0)
            else:
                log.warning("[NATSBus] No JetStream context; performing flush instead.")
                await self._safe_flush(timeout=2.0)
            return True
        except Exception as e:
            log.error("[NATSBus] Drain failed for %s: %s", full_subject, e)
            with contextlib.suppress(Exception):
                await self._safe_flush(timeout=2.0)
            return False

    def _start_keepalive(self):
        if self._keepalive_task and not self._keepalive_task.done():
            return

        async def _loop():
            try:
                while not self._stopping:
                    try:
                        if not self._connected or not self._nc or self._nc.is_closed:
                            await asyncio.sleep(3.0)
                            continue
                        await self._nc.flush(timeout=1.0)  # use NATS' own timeout
                        await asyncio.sleep(10.0 if self.debug else 20.0)
                    except (NatsTimeoutError, asyncio.TimeoutError):
                        self._connected = False
                        await asyncio.sleep(2.0)
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        self._connected = False
                        await asyncio.sleep(2.0)
            finally:
                pass

        keepalive_task = asyncio.create_task(_loop())
        self._tasks.add(keepalive_task)
        keepalive_task.add_done_callback(lambda f: self._tasks.discard(keepalive_task))
        self._keepalive_task = keepalive_task
        log.debug("NATSKeepaliveStarted")

    def _start_health_monitoring(self):
        if self._health_task and not self._health_task.done():
            return

        async def _health():
            try:
                while not self._stopping:
                    try:
                        if not self._stopping:
                            await asyncio.sleep(self.health_check_interval)
                        if not self._connected or not self._nc or self._nc.is_closed:
                            continue
                        await self._nc.flush(timeout=1.0)
                    except (NatsTimeoutError, asyncio.TimeoutError):
                        self._connected = False
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        self._connected = False
            finally:
                pass

        health_task = asyncio.create_task(_health())
        self._tasks.add(health_task)
        health_task.add_done_callback(lambda f: self._tasks.discard(health_task))
        self._health_task = health_task
        log.debug("NATSHealthMonitorStarted")

    async def _stop_tasks(self):
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
                try:
                    await asyncio.shield(t)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
        self._tasks.clear()

    def health_check(self) -> Dict[str, Any]:
        details = self.debug_connection_status()
        status = "connected" if (self._connected and self._nc and not self._nc.is_closed) else "disconnected"
        return {"bus_type": "nats", "status": status, "details": details}

    def _backoff(self, attempt: int) -> float:
        # jittered exponential backoff
        base = self.retry_base_delay * (2 ** attempt)
        return base * (1 + 0.2 * random.random())

    def _write_dlq(self, error: str, subject: str, envelope: dict) -> None:
        if not self._dlq_writer:
            return
        try:
            self._dlq_writer({"ts": time.time(), "error": error, "subject": subject, "envelope": envelope})
        except Exception as e:
            log.error("DLQWriterFailed subject: %s, error: %r", subject, e)

    async def _safe_flush(self, timeout: float = 1.0) -> bool:
        """
        Try to flush with NATS' own timeout; never raises CancelledError to callers.
        Returns True if OK, False if it timed out or errored.
        """
        if self._stopping or not self._nc or self._nc.is_closed:
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
            return (
                hasattr(sys, "gettrace") and sys.gettrace() is not None
                or "pydevd" in sys.modules
                or "pdb" in sys.modules
                or os.getenv("DEBUG", "0") == "1"
            )
        except Exception:
            return False


def jsonl_dlq_writer_factory(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def write(obj: dict):
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return write

def _backoff_gen(base=0.25, cap=5.0):
    delay = base
    while True:
        # 20% jitter
        yield min(cap, delay) * (0.8 + 0.4 * random.random())
        delay = min(cap, delay * 2.0)

async def _maybe_create_stream(js, name: str, subjects: List[str]):
    try:
        await js.stream_info(name)
        return
    except Exception:
        cfg = StreamConfig(
            name=name,
            subjects=subjects,
            retention=RetentionPolicy.LIMITS,
        )
        # If another process created it in between, this will raise; ignore.
        with contextlib.suppress(Exception):
            await js.add_stream(cfg)

async def _safe_flush(nc, timeout=2.0):
    try:
        await nc.flush(timeout=timeout)
    except (NatsTimeoutError, asyncio.TimeoutError):
        pass  # not fatal; connect succeeded but server slow to flush

async def _run_coro_fire_and_forget(coro):
    # keep reconnection callbacks lightweight
    asyncio.create_task(coro)