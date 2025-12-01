from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Dict, Optional

import zmq
import zmq.asyncio as azmq

log = logging.getLogger(__name__)


class ZmqKnowledgeBus:
    """
    BusProtocol-compatible adapter over ZeroMQ.

    Semantics:
      - publish(subject, payload): sends {"subject", "payload"} to workers
      - subscribe(subject, handler): spawns a worker-side socket and invokes handler(payload)
      - request(subject, payload): like publish but waits for reply matched by job_id

    Implementation:
      - Uses a DEALER socket (`client`) to talk to the broker frontend.
      - Uses a DEALER socket (`worker`) to talk to the broker backend when subscribe() is used.
      - No durable streams/consumers; ensure_* are no-ops that return True.
      - We embed the 'subject' in the JSON envelope so handlers can filter.
    """

    def __init__(
        self,
        fe_addr: str = "tcp://127.0.0.1:5555",
        be_addr: str = "tcp://127.0.0.1:5556",
        logger: Optional[logging.Logger] = None,
    ):
        self.request = None
        self.logger = logger or log
        self.ctx = azmq.Context.instance()
        self.fe_addr, self.be_addr = fe_addr, be_addr

        # client socket (DEALER) for publish/request
        self.client = self.ctx.socket(zmq.DEALER)     # client → frontend
        self.client.setsockopt(zmq.IDENTITY, uuid.uuid4().hex.encode())
        self.client.setsockopt(zmq.LINGER, 0)

        self._recv_task: Optional[asyncio.Task] = None
        self._pending: Dict[str, asyncio.Future] = {}

        # worker socket (DEALER) used when subscribe() is called
        self.worker = self.ctx.socket(zmq.DEALER)     # worker ← backend
        self.worker.setsockopt(zmq.IDENTITY, uuid.uuid4().hex.encode())
        self.worker.setsockopt(zmq.LINGER, 0)

        self._worker_task: Optional[asyncio.Task] = None
        self._subs: Dict[str, Callable[[Dict[str, Any]], Any]] = {}  # pattern -> handler

        self._connected = False

    # -------- lifecycle -------------------------------------------------------

    async def connect(self) -> bool:
        """
        Connect client side to the broker frontend and start the recv loop.
        Call this once early (e.g., in memory/bus init).
        """
        if self._connected:
            return True

        self.client.connect(self.fe_addr)
        self._recv_task = asyncio.create_task(self._client_recv_loop())
        self._connected = True
        return True

    async def close(self):
        """
        Cancel background tasks and close sockets.
        Safe to call multiple times.
        """
        # Cancel client recv loop
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            except Exception:
                self.logger.exception("ZmqKnowledgeBus: error stopping recv_task")
            self._recv_task = None

        # Cancel worker loop (if any)
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            except Exception:
                self.logger.exception("ZmqKnowledgeBus: error stopping worker_task")
            self._worker_task = None

        # Close sockets
        try:
            self.client.close(0)
        except Exception:
            pass
        try:
            self.worker.close(0)
        except Exception:
            pass

        # Fail any pending requests
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(asyncio.CancelledError())
        self._pending.clear()

        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -------- publish / request ----------------------------------------------

    async def publish(self, subject: str, payload: Dict[str, Any], **_: Any) -> None:
        """
        Fire-and-forget publish. Payload must be JSON-serializable.
        """
        env = {"subject": subject, "payload": payload}
        await self.client.send_json(env)

    async def publish_raw(self, subject: str, body: bytes, **_: Any) -> None:
        """
        Raw-ish path; we wrap raw bytes into a JSON struct so the bus shape stays consistent.
        """
        env = {
            "subject": subject,
            "payload": {"__binary__": True, "data_b64": body.decode("latin1")},
        }
        await self.client.send_json(env)

    async def request(
        self,
        subject: str,
        payload: Dict[str, Any],
        timeout: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Simple RPC over the bus:
          - attaches job_id to payload
          - waits for matching reply in _client_recv_loop
        """
        if not self._connected:
            await self.connect()

        job_id = payload.get("job_id") or uuid.uuid4().hex
        payload["job_id"] = job_id

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[job_id] = fut

        await self.publish(subject, payload)

        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(job_id, None)

    async def _client_recv_loop(self):
        """
        Client-side loop:
          - receives replies from broker frontend
          - matches on job_id to complete request() futures
        """
        while True:
            try:
                payload = await self.client.recv()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("ZMQ client recv error: %r", e)
                await asyncio.sleep(0.05)
                continue

            try:
                msg = json.loads(payload.decode("utf-8"))
                if not isinstance(msg, dict):
                    self.logger.warning(
                        "ZMQ client recv: non-dict message ignored: %r", msg
                    )
                    continue
                job_id = msg.get("job_id")
                if not job_id:
                    self.logger.warning(
                        "ZMQ client recv: missing job_id in %r", msg
                    )
                    continue
                fut = self._pending.pop(job_id, None)
                if fut and not fut.done():
                    fut.set_result(msg)
            except Exception as e:
                self.logger.error(
                    "ZMQ client decode error: %r (payload=%r)", e, payload[:128]
                )

    # -------- subscribe / worker side ----------------------------------------

    async def subscribe(
        self,
        subject: str,
        handler: Callable[[Dict[str, Any]], Any],
        **kwargs: Any,
    ):
        """
        Register a handler for a subject pattern (supports * and > like NATS).
        Handler always receives a dict (normalized by _coerce_body_to_dict).
        """
        self._subs[subject] = handler

        # Lazily attach worker to backend and start loop
        if not self._worker_task:
            self.worker.connect(self.be_addr)
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self):
        """
        Worker-side loop:
          - receives envelopes from broker backend
          - finds matching handler via subject patterns
          - normalizes body to dict
          - routes:
              * results.* → one-way delivery, no reply or re-publish
              * others:
                  - if body.result_subject → publish fanout result
                  - else → RPC reply to client
        """
        while True:
            try:
                client_id, payload = await self.worker.recv_multipart()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("ZMQ worker recv error: %r", e)
                await asyncio.sleep(0.05)
                continue

            try:
                env = json.loads(payload.decode("utf-8"))
            except Exception:
                self.logger.error("ZMQ worker decode error: %r", payload[:128])
                continue

            subj = env.get("subject", "")
            body_raw = env.get("payload", {})
            body = _coerce_body_to_dict(body_raw)
            is_results = isinstance(subj, str) and subj.startswith("results.")

            # wildcard match
            handler = None
            for pattern, h in list(self._subs.items()):
                if _match_subject(pattern, subj):
                    handler = h
                    break

            if not handler:
                # optionally log unknown subject
                # self.logger.debug("No handler for subject=%s", subj)
                continue

            # run user handler (always pass normalized dict)
            try:
                maybe = handler(body)
                resp = await maybe if asyncio.iscoroutine(maybe) else maybe
            except Exception as e:
                self.logger.exception("ZMQ handler error for subject=%s", subj)
                resp = {
                    "job_id": body.get("job_id"),
                    "status": "error",
                    "error": str(e),
                }

            # ----- One-way result events: never RPC-reply or re-publish -----
            # We deliver to subscribers via the handler above; nothing else to do.
            if is_results:
                continue

            # ----- Submit path: publish to requested results subject (NATS-style fanout) -----
            result_subject = body.get("result_subject") or env.get("result_subject")
            if result_subject and isinstance(resp, dict):
                await self.publish(result_subject, resp)
                # do NOT also RPC reply
                continue

            # ----- RPC: reply only if the handler returned a dict -----
            if isinstance(resp, dict):
                if body.get("job_id"):
                    resp.setdefault("job_id", body.get("job_id"))
                await self.worker.send_multipart(
                    [client_id, json.dumps(resp).encode("utf-8")]
                )
            # else: handler returned None → no reply (by design)

    def wrap_request(self, middleware):
        """
        Install a middleware that wraps the low-level request call.

        middleware(next_handler, subject, payload, **kwargs) -> Any
        """
        # Save original
        base_request = self.request

        async def wrapped(subject: str, payload: Dict[str, Any], timeout: float = 5.0):
            return await middleware(
                base_request,
                subject,
                payload,
                method="request",
                timeout=timeout,
            )

        self.request = wrapped

    # -------- JetStream-ish no-ops -------------------------------------------

    async def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._connected

    async def ensure_stream(self, stream: str, subjects: list[str]) -> bool:
        return True

    async def ensure_consumer(self, **kwargs) -> bool:
        return True

    async def flush(self, timeout: float = 1.0) -> bool:
        return True

    async def drain_subject(self, subject: str) -> bool:
        return True

    # Debug helper to mirror Hybrid health
    def debug_connection_status(self) -> Dict[str, Any]:
        return {
            "connection_uptime": 0,
            "reconnect_attempts": 0,
            "debug_mode": False,
        }


# ---------- helpers ----------------------------------------------------------


def _match_subject(pattern: str, subject: str) -> bool:
    """
    NATS-style subject match:
      - exact tokens
      - * matches a single token
      - > matches the rest
    """
    if pattern == subject:
        return True
    pp = pattern.split(".")
    ss = subject.split(".")
    i = 0
    while i < len(pp) and i < len(ss):
        if pp[i] == ">":
            return True
        if pp[i] != "*" and pp[i] != ss[i]:
            return False
        i += 1
    # exact end or trailing '>' at pattern end matches
    return i == len(pp) == len(ss) or (i == len(pp) - 1 and pp[-1] == ">")


def _coerce_body_to_dict(x: Any) -> Dict[str, Any]:
    """
    Normalize body to a dict so .get() is always safe.
    If decoding fails, returns {"_raw": "<...>"}.
    """
    try:
        if isinstance(x, (bytes, bytearray)):
            try:
                return json.loads(x.decode("utf-8"))
            except Exception:
                return {"_raw": x.decode("utf-8", "replace")}
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return {"_raw": x}
        if isinstance(x, dict):
            return x
        # Last resort: represent as string
        return {"_raw": str(x)}
    except Exception:
        return {"_raw": "<decode_error>"}
