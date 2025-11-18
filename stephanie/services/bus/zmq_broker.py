# stephanie/services/bus/zmq_broker.py
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from contextlib import suppress

import zmq
import zmq.asyncio as azmq

log = logging.getLogger(__name__)


class ZMQBroker:
    _instance: "ZMQBroker | None" = None

    def __init__(
        self,
        fe: str = "tcp://127.0.0.1:5555",
        be: str = "tcp://127.0.0.1:5556",
        idle_seconds: int | None = None,
    ):
        self.ctx = azmq.Context.instance()
        self.fe_addr, self.be_addr = fe, be
        self.front = self.ctx.socket(zmq.ROUTER)
        self.back = self.ctx.socket(zmq.DEALER)

        # Ensure fast teardown
        self.front.setsockopt(zmq.LINGER, 0)
        self.back.setsockopt(zmq.LINGER, 0)

        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        self._closing = False

        # Idle shutdown support
        self._idle_seconds: int | None = idle_seconds
        self._last_activity_ts: float | None = None

    # ---------- lifecycle -----------------------------------------------------

    @classmethod
    async def start(
        cls,
        fe: str = "tcp://127.0.0.1:5555",
        be: str = "tcp://127.0.0.1:5556",
        idle_seconds: int | None = None,
    ):
        """
        Create (or return) singleton and start background loop.

        idle_seconds:
          - if not None, broker will auto-shutdown after that many seconds
            of no traffic across front/back sockets.
        """
        if cls._instance:
            # Optionally update idle window on an existing broker
            broker = cls._instance
            broker._idle_seconds = idle_seconds
            return broker

        broker = cls(fe, be, idle_seconds=idle_seconds)
        cls._instance = broker
        broker._task = asyncio.create_task(broker._run(), name="ZMQBroker.run")
        return broker

    @classmethod
    async def close(cls):
        """Gracefully stop the broker and clean up resources."""
        broker = cls._instance
        if not broker:
            return

        broker._closing = True
        broker._stop_evt.set()  # release hold()

        # Wait for run task to finish without surfacing CancelledError
        cur = asyncio.current_task()
        if broker._task and broker._task is not cur:
            with suppress(asyncio.CancelledError):
                await broker._task

        # Sockets/context cleanup
        with suppress(Exception):
            broker.front.close(0)
        with suppress(Exception):
            broker.back.close(0)
        # Do NOT terminate the shared Context.instance() here; others may use it.

        broker._task = None
        cls._instance = None

    # ---------- run loop ------------------------------------------------------

    async def _run(self):
        try:
            # Bind sockets once here
            self.front.bind(self.fe_addr)
            self.back.bind(self.be_addr)

            poller = azmq.Poller()
            poller.register(self.front, zmq.POLLIN)
            poller.register(self.back, zmq.POLLIN)

            loop = asyncio.get_running_loop()

            while not self._stop_evt.is_set():
                try:
                    # Finite timeout so loop reacts quickly to stop
                    events = dict(await poller.poll(timeout=200))  # ms
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.warning("ZMQBroker poll error: %s", e)
                    await asyncio.sleep(0.05)
                    continue

                had_traffic = False

                if self.front in events:
                    # Client → Worker : [client_id, payload]
                    frames = await self.front.recv_multipart()
                    await self.back.send_multipart(frames)
                    had_traffic = True

                if self.back in events:
                    # Worker → Client : [client_id, payload]
                    frames = await self.back.recv_multipart()
                    await self.front.send_multipart(frames)
                    had_traffic = True

                if had_traffic:
                    self._last_activity_ts = loop.time()

                # Optional idle auto-shutdown
                if (
                    self._idle_seconds is not None
                    and self._last_activity_ts is not None
                ):
                    now = loop.time()
                    if now - self._last_activity_ts > float(self._idle_seconds):
                        log.info(
                            "ZMQBroker idle timeout (%ss) reached; shutting down",
                            self._idle_seconds,
                        )
                        break

        except asyncio.CancelledError:
            # normal during shutdown
            pass
        finally:
            self._closing = True
            self._stop_evt.set()
            # Best-effort close here as well; close() will repeat safely
            with suppress(Exception):
                self.front.close(0)
            with suppress(Exception):
                self.back.close(0)

    # ---------- foreground wait ----------------------------------------------

    @classmethod
    async def hold(cls):
        """Optional foreground wait: returns when broker is asked to close."""
        if not cls._instance:
            return
        try:
            await cls._instance._stop_evt.wait()
        except asyncio.CancelledError:
            pass


# --------------------------------------------------------------------------- #
# Singleton guard to ensure broker is up (and optionally managed)
# --------------------------------------------------------------------------- #

class ZmqBrokerGuard:
    """
    Starts the broker once and (optionally) keeps it up across pipelines.

    Modes:
      - detached=True (default): pipelines never close the broker (close() no-op).
      - detached=False (managed): pipeline that started it will close on guard.close().
      - idle_seconds: broker auto-shuts down after N seconds of inactivity (optional).

    Env overrides:
      - STEPH_BROKER_FE       (default tcp://127.0.0.1:5555)
      - STEPH_BROKER_BE       (default tcp://127.0.0.1:5556)
      - STEPH_BROKER_MODE     ("detached" | "managed")  [default: detached]
      - STEPH_BROKER_IDLE_S   (integer seconds, optional)
    """

    _instance_task: asyncio.Task | None = None
    _detached: bool = True
    _idle_seconds: int | None = None
    _fe: str = "tcp://127.0.0.1:5555"
    _be: str = "tcp://127.0.0.1:5556"

    @classmethod
    def _load_env_defaults(cls):
        cls._fe = os.getenv("STEPH_BROKER_FE", cls._fe)
        cls._be = os.getenv("STEPH_BROKER_BE", cls._be)

        # STEPH_BROKER_MODE: "detached" | "managed"
        mode = os.getenv("STEPH_BROKER_MODE", "detached").strip().lower()
        cls._detached = mode != "managed"

        # Optional idle timeout (seconds)
        idle = os.getenv("STEPH_BROKER_IDLE_S", "30000").strip()
        cls._idle_seconds = int(idle) if idle.isdigit() and int(idle) > 0 else None

    @classmethod
    async def ensure_started(
        cls,
        fe: str | None = None,
        be: str | None = None,
        *,
        detached: bool | None = None,
        idle_seconds: int | None = None,
    ):
        """
        Start the singleton broker if not running.
        - detached: True → pipelines must NOT close broker (default).
        - idle_seconds: optional inactivity window for auto-shutdown.
        """
        cls._load_env_defaults()

        if fe:
            cls._fe = fe
        if be:
            cls._be = be
        if detached is not None:
            cls._detached = bool(detached)
        if idle_seconds is not None:
            cls._idle_seconds = idle_seconds

        # Start (or update) the broker
        await ZMQBroker.start(cls._fe, cls._be, idle_seconds=cls._idle_seconds)

        # Only create a hold task if we’re managing lifecycle (non-detached).
        if not cls._detached:
            if cls._instance_task and not cls._instance_task.done():
                return

            async def _hold():
                try:
                    await ZMQBroker.hold()
                finally:
                    # In managed mode, we own shutdown
                    with contextlib.suppress(Exception):
                        await ZMQBroker.close()

            cls._instance_task = asyncio.create_task(
                _hold(), name="ZMQBrokerGuard.hold"
            )

    @classmethod
    async def close(cls, *, force: bool = False):
        """
        Close the broker if:
          - managed mode (detached=False), or
          - force=True explicitly overrides detached mode.
        """
        if cls._detached and not force:
            # No-op in detached mode
            return

        with suppress(Exception):
            await ZMQBroker.close()

        t = cls._instance_task
        cls._instance_task = None
        if t:
            with suppress(asyncio.CancelledError):
                await t
