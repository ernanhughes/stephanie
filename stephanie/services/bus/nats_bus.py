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
from nats.js.api import (ConsumerConfig, DeliverPolicy, RetentionPolicy,
                         StreamConfig)

from .bus_protocol import BusProtocol
from .errors import BusRequestError
from .idempotency import InMemoryIdempotencyStore

_logger = logging.getLogger(__name__)


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
        timeout: float = 1.0,  # publish/request deadline (s)
        max_retries: int = 3,
        retry_base_delay: float = 0.2,  # backoff base
        max_in_flight: int = 256,  # bound concurrent publishes
        health_check_interval: float = 30.0,
        debug: bool = False,  # stickier settings for debug
        fire_and_forget_subjects: Optional[
            Set[str]
        ] = None,  # subjects to bypass JS
        dlq_writer: Optional[
            Callable[[dict], None]
        ] = None,  # optional JSONL writer
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
        self._idem_store = None
        self._connected = False

        self._sem = asyncio.Semaphore(max_in_flight)
        self._health_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

        # subject -> {handler, durable, cb, sub}
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._last_publish_time = 0.0
        self._publish_failures = 0
        self._dlq_writer = dlq_writer

        # Subjects that should use plain NATS (no PUBACK, no JS), useful for telemetry
        self._faf_subjects: Set[str] = fire_and_forget_subjects or set()
        self._tasks: Set[asyncio.Task] = set()

        # Connection guards
        self._conn_lock = asyncio.Lock()
        self._conn_task: Optional[asyncio.Task] = None

        # Stream subjects we ensure; default to "<stream>.>"
        self._stream_subjects: List[str] = [f"{self.stream}.>"]
        self._stopping: bool = False

    # ---------- Connection management ----------

    async def connect(self, force: bool = False) -> bool:
        """
        Connects to NATS and ensures JetStream stream is present.
        Returns True only after:
          1) TCP connected
          2) JetStream context acquired
          3) Stream exists or is created with subjects
          4) Health publish succeeds
        On failure, closes the temp connection and returns False.
        """
        # Fast-path when already connected and not forcing
        if (
            self._connected
            and self._nc
            and not self._nc.is_closed
            and not force
        ):
            return True

        async with self._conn_lock:
            if (
                self._connected
                and self._nc
                and not self._nc.is_closed
                and not force
            ):
                return True

            # If forcing, drain old connection
            if self._nc and not self._nc.is_closed and force:
                with contextlib.suppress(Exception):
                    await self._nc.drain()

            # Establish a fresh connection and only commit on full readiness
            import nats

            async def _err_cb(e):
                # nats InvalidStateError on pong is harmless; low-level
                _logger.debug(f"NATSLowLevelError error: {repr(e)}")

            async def _disc_cb():
                _logger.debug("NATSDisconnected")

            async def _reconn_cb():
                _logger.debug("NATSReconnected")
                await self._on_reconnected()

            async def _closed_cb():
                _logger.debug("NATSClosed")

            nc = None
            try:
                nc = await nats.connect(
                    servers=self.servers,
                    name=self.stream,
                    allow_reconnect=True,
                    max_reconnect_attempts=-1,
                    reconnect_time_wait=0.5,  # quicker retries
                    ping_interval=5,  # more frequent pings
                    max_outstanding_pings=2,  # fail fast if peer unresponsive
                    error_cb=_err_cb,
                    disconnected_cb=_disc_cb,
                    reconnected_cb=_reconn_cb,
                    closed_cb=_closed_cb,
                )
                js = nc.jetstream()

                # Ensure stream exists; create if missing with "<stream>.>" subjects
                try:
                    await js.stream_info(self.stream)
                except Exception:
                    cfg = StreamConfig(
                        name=self.stream,
                        subjects=self._stream_subjects,
                        retention=RetentionPolicy.Limits,
                    )
                    await js.add_stream(cfg)

                # Health publish to confirm JS perms
                health_subject = f"{self.stream}.health"
                try:
                    await js.publish(health_subject, b"ping")
                except Exception as e:
                    _logger.error(
                        "NATSHealthPublishFailed subject: %s, error: %r",
                        health_subject,
                        e,
                    )
                    with contextlib.suppress(Exception):
                        await nc.close()
                    return False

                # Commit only now
                self._nc = nc
                self._js = js
                self._connected = True
                _logger.debug(
                    "NATSReady servers: %s, stream: %s, subjects: %s",
                    self.servers,
                    self.stream,
                    self._stream_subjects,
                )

                # Start monitors (idempotent)
                self._start_keepalive()
                self._start_health_monitoring()
                return True

            except Exception as e:
                _logger.error("NATSClientConnectFailed error: %r", e)
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
            _logger.error("NATSRefreshJSOnReconnectFailed error: %r", e)
        await self._resubscribe_all()
        _logger.debug("NATSReconnectedSubscriptionsRefreshed")

    async def _ensure_connected(self) -> None:
        if self._connected and self._nc and not self._nc.is_closed:
            return
        ok = await self.connect()
        if not ok:
            # Let caller handle; do not raise here to keep behavior consistent
            _logger.error("NATSEnsureConnectFailed")

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
                # Older nats-py: no headers support
                if self.logger:
                    self.logger.debug("NATSNoHeaderSupportCore", {"subject": subject})
        await self._nc.publish(subject, data)

    async def _try_js_publish(self, subject: str, data: bytes, headers: Optional[Dict[str, str]]) -> None:
        # JetStream publish with optional headers; fallback for older clients
        if headers:
            try:
                await self._js.publish(subject, data, headers=headers)  # type: ignore[call-arg]
                return
            except TypeError:
                if self.logger:
                    self.logger.debug("NATSNoHeaderSupportJS", {"subject": subject})
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
            _logger.error("NATSPublishSkippedNotConnected subject: %s", subject)
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
                _logger.error("NATSPublishErrorRetrying subject: %s, error: %r", subject, e)
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
            _logger.error("NATSPublishRawSkippedNotConnected subject: %s", subject)
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
                _logger.error("NATSPublishRawErrorRetrying subject: %s, error: %r", subject, e)
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
            _logger.error(
                "NATSRequestSkippedNotConnected subject: %s", subject
            )
            return None

        request_timeout = min(timeout, self.timeout)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        rpc_subject = f"{self.stream}.rpc.{subject}"

        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._nc.request(
                    rpc_subject, data, timeout=request_timeout
                )
                return json.loads(resp.data.decode())
            except NatsTimeoutError:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                _logger.error(
                    "NATSRequestTimeout subject: %s, attempts: %d",
                    subject,
                    attempt + 1,
                )
                return None
            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                _logger.error(
                    "NATSRequestFailed subject: %s, error: %r", subject, e
                )
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
            _logger.error("NATSSubscribeSkippedNotConnected subject: %s", subject)
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
        ack_td = _as_timedelta(ack_wait, default_seconds=(self.timeout * 5))
        cfg = ConsumerConfig(
            deliver_policy=deliver_policy or DeliverPolicy.ALL,
            ack_wait=ack_td,
            max_deliver=(self.max_retries + 1) if max_deliver is None else int(max_deliver),
        )
        full_subject = f"{self.stream}.{subject}"

        # prefer passing queue if supported; otherwise fall back
        try:
            sub = await self._js.subscribe(             # type: ignore[call-arg]
                full_subject,
                durable=durable_name,
                cb=cb,
                config=cfg,
                queue=queue,                            # <- correct kw for nats-py
            )
        except TypeError:
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
                _logger.debug("NATSResubscribed subject: %s", subject)
            except Exception as e:
                _logger.error("NATSResubscribeFailed subject: %s, error: %r", subject, e)

    # ---------- Shutdown ----------

    async def stop(self) -> None:
        """Cooperative shutdown: stop loops first, then drain/close NATS."""
        self._stopping = True
        await self._stop_tasks()  # cancels keepalive/health tasks
        # Unsubscribe + drain/close like close(), but protected for shutdown races
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
                # Unsubscribe cleanly
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
                _logger.debug("NATSClosedCleanly")

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
                if self._last_publish_time
                else 0
            ),
            "debug_mode": self.debug,
            "timeout": self.timeout,
            "reconnect_attempts": self.max_retries,
            "keepalive_interval": 10.0 if self.debug else 20.0,
            "health_check_interval": self.health_check_interval,
            "subscriptions": list(self._subscriptions.keys()),
        }

    def _build_wrapped(
        self, subject: str, handler: Callable[[Dict[str, Any]], Any]
    ):
        async def wrapped(msg):
            try:
                # Try JSON first; if it fails, deliver raw bytes to handler.
                try:
                    envelope = json.loads(msg.data.decode())
                except Exception:
                    # Pass raw bytes to consumer (iterator/callback may handle bytes).
                    await handler(msg.data)
                    await msg.ack()
                    return

                # Idempotency (best-effort if event_id exists)
                event_id = envelope.get("event_id")
                if event_id and await self.idempotency_store.seen(event_id):
                    await msg.ack()
                    self.logger.debug(
                        "NATSDuplicateEventSkipped", {"event_id": event_id}
                    )
                    return
                if event_id:
                    await self.idempotency_store.mark(event_id)

                # Support both styles: raw dict, or {"payload": {...}}
                payload = envelope.get("payload", envelope)

                await handler(payload)
            except Exception as e:
                self.logger.error(
                    "NATSHandlerError", {"subject": subject, "error": repr(e)}
                )
            finally:
                with contextlib.suppress(Exception):
                    await msg.ack()

        return wrapped

    # Low-level NATS callbacks (not used directly; we bind lambdas in connect)
    async def _on_disconnected_cb(self, *args, **kwargs):
        _logger.error("NATSDisconnected")

    async def _on_reconnected_cb(self, *args, **kwargs):
        try:
            self._js = self._nc.jetstream()
        except Exception as e:
            _logger.error(f"NATSRefreshJSOnReconnectFailed error: {repr(e)}")
        await self._resubscribe_all()
        _logger.debug("NATSReconnectedSubscriptionsRefreshed")

    async def _on_closed_cb(self, *args, **kwargs):
        _logger.error("NATSClosed")

    async def _on_error_cb(self, e):
        _logger.error(f"NATSLowLevelError error: {repr(e)}")

    # Monitors

    def _start_keepalive(self):
        if self._keepalive_task and not self._keepalive_task.done():
            return

        async def _loop():
            try:
                while not self._stopping:
                    try:
                        if (
                            not self._connected
                            or not self._nc
                            or self._nc.is_closed
                        ):
                            await asyncio.sleep(3.0)
                            continue
                        # Use NATS' own timeout; no external wait_for
                        await self._nc.flush(timeout=1.0)
                        await asyncio.sleep(10.0 if self.debug else 20.0)

                    except (NatsTimeoutError, asyncio.TimeoutError):
                        # soft mark as unhealthy; next op will reconnect
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
        keepalive_task.add_done_callback(
            lambda f: self._tasks.discard(keepalive_task)
        )
        self._keepalive_task = keepalive_task
        _logger.debug("NATSKeepaliveStarted")

    # ---------- Flush / Drain Helpers ----------

    async def flush(self, timeout: float = 1.0) -> bool:
        """
        Wait for all pending messages and PUBACKs to be processed.
        Returns True if successful, False on timeout or error.
        """
        return await self._safe_flush(timeout=timeout)

    async def drain_subject(self, subject: str) -> bool:
        """
        Drain (purge or flush) pending JetStream messages for a subject.
        Works across all nats-py versions.
        """
        await self._ensure_connected()
        full_subject = f"{self.stream}.{subject}"
        try:
            # Prefer a safe flush if stream isn't large — prevents unwanted data loss
            if self._js:
                try:
                    # Purge entire stream (subject filters not universally supported)
                    await self._js.purge_stream(self.stream)
                    _logger.debug(
                        "[NATSBus] Stream '%s' purged (no subject filter).",
                        self.stream,
                    )
                except TypeError:
                    # In some versions, purge_stream is not on context; fall back to flush
                    _logger.warning(
                        "[NATSBus] purge_stream() unsupported; flushing %s.",
                        full_subject,
                    )
                    await self._safe_flush(timeout=2.0)
            else:
                _logger.warning(
                    "[NATSBus] No JetStream context; performing flush instead."
                )
                await self._safe_flush(timeout=2.0)

            return True

        except Exception as e:
            _logger.error("[NATSBus] Drain failed for %s: %s", full_subject, e)
            try:
                await self._safe_flush(timeout=2.0)
            except Exception:
                pass
            return False

    def _start_health_monitoring(self):
        if self._health_task and not self._health_task.done():
            return

        async def _health():
            try:
                while not self._stopping:
                    try:
                        if not self._stopping:
                            await asyncio.sleep(self.health_check_interval)
                        if (
                            not self._connected
                            or not self._nc
                            or self._nc.is_closed
                        ):
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
        health_task.add_done_callback(
            lambda f: self._tasks.discard(health_task)
        )
        self._health_task = health_task
        _logger.debug("NATSHealthMonitorStarted")

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
        status = (
            "connected"
            if (self._connected and self._nc and not self._nc.is_closed)
            else "disconnected"
        )
        return {"bus_type": "nats", "status": status, "details": details}

    def _backoff(self, attempt: int) -> float:
        # jittered exponential backoff
        base = self.retry_base_delay * (2**attempt)
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
        except Exception as e:
            _logger.error("DLQWriterFailed subject: %s, error: %r", subject, e)

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
                hasattr(sys, "gettrace")
                and sys.gettrace() is not None
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
