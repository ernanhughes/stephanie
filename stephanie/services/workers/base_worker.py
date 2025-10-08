from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

_logger = logging.getLogger(__name__)

Handler = Callable[[Dict[str, Any]], Awaitable[None]]



class BaseWorker:
    """
    A reusable foundation for bus-backed workers.

    Responsibilities:
      - Ensure bus connection (idempotent)
      - Subscribe with retries + exponential backoff
      - Periodic health checks (bus + worker-specific)
      - Graceful startup/shutdown
      - Structured logging
      - Service access via container
    """

    MAX_SUB_RETRIES: int = 5
    SUB_RETRY_BASE_SEC: float = 1.0
    HEALTH_INTERVAL_SEC: float = 30.0

    def __init__(
        self,
        *,
        name: str,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,                    
        logger,
    ):
        self.name = name
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container         
        self.logger = logger

        self._stop_evt = asyncio.Event()
        self._health_task: Optional[asyncio.Task] = None
        self._subscriptions: Dict[str, Handler] = {}
        self._sub_retry_counts: Dict[str, int] = {}
        self._started = False

        self._stats = {
            "started_at": None,
            "messages_in": 0,
            "messages_err": 0,
            "subscriptions": 0,
            "last_health_ok": 0.0,
        }

        wcfg = self.cfg.get("worker", {})
        self.max_retries = int(wcfg.get("max_sub_retries", self.MAX_SUB_RETRIES))
        self.retry_base = float(wcfg.get("sub_retry_base_sec", self.SUB_RETRY_BASE_SEC))
        self.health_interval = float(wcfg.get("health_interval_sec", self.HEALTH_INTERVAL_SEC))

    # --------------------------- lifecycle --------------------------- #

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stats["started_at"] = time.time()

        # 0) allow subclasses to init/resolve services from container
        try:
            await self.init_services()
        except Exception as e:
            _logger.error("WorkerInitServicesFailed: name=%s, error=%s", self.name, str(e))

        # 1) ensure bus is up
        try:
            await self.memory.ensure_bus_connected()
        except Exception as e:
            _logger.info("WorkerBusConnectFailed: name=%s, error=%s", self.name, str(e))

        # 2) subclass registers subjects
        await self.register_subjects()

        # 3) start health loop
        self._health_task = asyncio.create_task(self._health_loop())
        _logger.info("WorkerStarted: name=%s, subs=%s", self.name, list(self._subscriptions.keys()))

    async def stop(self) -> None:
        if not self._started:
            return
        self._stop_evt.set()
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._health_task
        self._started = False
        _logger.info("WorkerStopped: name=%s", self.name)

    # ---------------------- subclass extension points ---------------------- #

    async def init_services(self) -> None:
        """Optional: resolve/initialize services from the container."""
        return

    async def register_subjects(self) -> None:
        """Subclasses call self.subscribe(subject, handler) here."""
        raise NotImplementedError

    async def worker_health(self) -> Dict[str, Any]:
        """Optional: subclasses return extra health metrics."""
        return {}

    # ----------------------- container helpers ----------------------- #

    def get_service(self, key: str) -> Any:
        """Fetch a service from the container (may return None)."""
        getter = getattr(self.container, "get", None)
        if callable(getter):
            return getter(key)
        # fallback: dict-like
        try:
            return self.container[key]
        except Exception:
            return None

    def require_service(self, key: str) -> Any:
        """Fetch a service or raise a clear error."""
        svc = self.get_service(key)
        if svc is None:
            raise RuntimeError(f"Service '{key}' not found in container for worker '{self.name}'.")
        return svc

    # ----------------------- bus helpers (public) ----------------------- #

    async def subscribe(self, subject: str, handler: Handler) -> None:
        self._subscriptions[subject] = handler
        self._sub_retry_counts.setdefault(subject, 0)
        await self._subscribe_with_retry(subject, handler)

    async def publish(self, subject: str, payload: Dict[str, Any]) -> None:
        try:
            await self.memory.bus.publish(subject=subject, payload=payload)
        except Exception as e:
            _logger.info("WorkerPublishError: name=%s, subject=%s, error=%s", self.name, subject, str(e))
            raise

    # --------------------------- internals --------------------------- #

    async def _subscribe_with_retry(self, subject: str, handler: Handler) -> None:
        i = self._sub_retry_counts.get(subject, 0)
        try:
            await self.memory.bus.subscribe(subject, self._wrap_handler(handler))
            self._sub_retry_counts[subject] = 0
            self._stats["subscriptions"] += 1
            _logger.info("WorkerSubscribed: name=%s, subject=%s", self.name, subject)
        except Exception as e:
            i += 1
            self._sub_retry_counts[subject] = i
            if i <= self.max_retries:
                delay = self.retry_base * (2 ** (i - 1))
                _logger.info("WorkerSubscribeRetry: name=%s, subject=%s, retry=%d, delay_sec=%d, error=%s", self.name, subject, i, delay, str(e))
                await asyncio.sleep(delay)
                await self._subscribe_with_retry(subject, handler)
            else:
                _logger.info("WorkerSubscribeFailed: name=%s, subject=%s, error=%s", self.name, subject, str(e))

    def _wrap_handler(self, handler: Handler) -> Handler:
        async def wrapped(payload: Dict[str, Any]):
            self._stats["messages_in"] += 1
            try:
                await handler(payload)
            except Exception as e:
                self._stats["messages_err"] += 1
                _logger.info("WorkerHandlerError name=%s, error=%s", self.name, str(e))
        return wrapped

    async def _health_loop(self):
        while not self._stop_evt.is_set():
            try:
                # bus health
                try:
                    bus_health = self.memory.bus.health_check()
                except Exception as e:
                    bus_health = {"status": "unknown", "error": str(e)}

                worker_health = await self.worker_health()
                payload = {
                    "name": self.name,
                    "bus": bus_health,
                    "worker": worker_health,
                    "stats": dict(self._stats),
                    "ts": time.time(),
                }
                _logger.info("WorkerHealth: name=%s", self.name)
                self._stats["last_health_ok"] = time.time()
            except Exception as e:
                _logger.info("WorkerHealthError: name=%s, error=%s", self.name, str(e))
            await asyncio.sleep(self.health_interval)

    def _log(self, event: str, data: Dict[str, Any]):
        if self.logger and hasattr(self.logger, "log"):
            self.logger.log(event, data)
        else:
            print(f"[{event}] {data}")
