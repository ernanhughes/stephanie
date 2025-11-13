from __future__ import annotations
import asyncio
import logging
import contextlib
import zmq
import zmq.asyncio as azmq

log = logging.getLogger(__name__)


class ZmqBroker:
    def __init__(self, fe="tcp://127.0.0.1:5555", be="tcp://127.0.0.1:5556"):
        self.ctx = azmq.Context.instance()
        self.fe_addr, self.be_addr = fe, be
        self.front = self.ctx.socket(zmq.ROUTER)  # unchanged
        self.back = self.ctx.socket(
            zmq.DEALER
        )  # <-- was ROUTER; make it DEALER
        self._task = None

    async def start(self):
        self.front.bind(self.fe_addr)
        self.back.bind(self.be_addr)
        log.info("ZMQ broker started FE=%s  BE=%s", self.fe_addr, self.be_addr)
        self._task = asyncio.create_task(self._run())

    async def close(self):
        # stop poll loop
        if self._task:
            self._task.cancel()
            with contextlib.suppress(Exception):
                await self._task
            self._task = None
        # close sockets
        with contextlib.suppress(Exception):
            self.front.close(0)
        with contextlib.suppress(Exception):
            self.back.close(0)

    async def _run(self):
        poller = azmq.Poller()
        poller.register(self.front, zmq.POLLIN)
        poller.register(self.back, zmq.POLLIN)

        while True:
            evts = dict(await poller.poll())

            # Client → Worker : [client_id, payload] (2 frames)
            if self.front in evts:
                frames = (
                    await self.front.recv_multipart()
                )  # [client_id, payload]
                # forward to any worker (DEALER handles fair-queueing)
                await self.back.send_multipart(frames)

            # Worker → Client : [client_id, payload] (2 frames)
            if self.back in evts:
                frames = (
                    await self.back.recv_multipart()
                )  # [client_id, payload]
                await self.front.send_multipart(
                    frames
                )  # ROUTER uses client_id to route


# Singleton “ensure it’s up” helper ------------------------------------------
class ZmqBrokerGuard:
    _instance_task: asyncio.Task | None = None
    _instance: ZmqBroker | None = None

    @classmethod
    async def ensure_started(
        cls, fe="tcp://127.0.0.1:5555", be="tcp://127.0.0.1:5556"
    ):
        if cls._instance_task and not cls._instance_task.done():
            return
        broker = ZmqBroker(fe, be)
        await broker.start()
        cls._instance = broker

        async def _hold():
            try:
                await asyncio.Event().wait()
            finally:
                if cls._instance:
                    with contextlib.suppress(Exception):
                        await cls._instance.close()
                cls._instance = None

        cls._instance_task = asyncio.create_task(_hold())
