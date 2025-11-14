# stephanie/services/bus/zmq_broker.py
from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import contextlib
import zmq
import zmq.asyncio as azmq

log = logging.getLogger(__name__)

class ZMQBroker:
    _instance: "ZMQBroker | None" = None

    def __init__(self, fe="tcp://127.0.0.1:5555", be="tcp://127.0.0.1:5556"):
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

    # ---------- lifecycle -----------------------------------------------------

    @classmethod
    async def start(cls, fe="tcp://127.0.0.1:5555", be="tcp://127.0.0.1:5556"):
        """Create (or return) singleton and start background loop."""
        if cls._instance:
            return cls._instance
        broker = cls(fe, be)
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

                if self.front in events:
                    # Client → Worker : [client_id, payload]
                    frames = await self.front.recv_multipart()
                    await self.back.send_multipart(frames)

                if self.back in events:
                    # Worker → Client : [client_id, payload]
                    frames = await self.back.recv_multipart()
                    await self.front.send_multipart(frames)

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


# Singleton guard to ensure broker is up (optional)
class ZmqBrokerGuard:
    _instance_task: asyncio.Task | None = None

    @classmethod
    async def ensure_started(cls, fe="tcp://127.0.0.1:5555", be="tcp://127.0.0.1:5556"):
        if cls._instance_task and not cls._instance_task.done():
            return
        # Start the singleton with provided addresses
        await ZMQBroker.start(fe, be)

        async def _hold():
            try:
                await ZMQBroker.hold()
            finally:
                with contextlib.suppress(Exception):
                    await ZMQBroker.close()

        cls._instance_task = asyncio.create_task(_hold(), name="ZMQBrokerGuard.hold")

    @classmethod
    async def close(cls):
        with suppress(Exception):
            await ZMQBroker.close()
        t = cls._instance_task
        cls._instance_task = None
        if t:
            with suppress(asyncio.CancelledError):
                await t
