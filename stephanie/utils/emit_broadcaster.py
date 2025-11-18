# stephanie/utils/emit_broadcaster.py
from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

SinkFn = Callable[[str, Dict[str, Any]], Any]  # may return Awaitable[None] or None
FilterFn = Callable[[str, Dict[str, Any]], bool]
ErrorFn = Callable[[str, Dict[str, Any], BaseException, str], None]
HookFn = Callable[[str, Dict[str, Any]], None]


@dataclass
class SinkSpec:
    fn: SinkFn
    name: str = "sink"
    only: Optional[Set[str]] = None           # if provided, allow only these events
    skip: Optional[Set[str]] = None           # if provided, block these events
    filter_fn: Optional[FilterFn] = None      # custom predicate(event, payload) -> bool
    max_concurrency: int = 64                 # per-sink concurrency bound
    _sem: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self):
        self._sem = asyncio.Semaphore(max(1, int(self.max_concurrency)))

    def allows(self, event: str, payload: Dict[str, Any]) -> bool:
        if self.only is not None and event not in self.only:
            return False
        if self.skip is not None and event in self.skip:
            return False
        if self.filter_fn is not None and not self.filter_fn(event, payload):
            return False
        return True


class EmitBroadcaster:
    """
    Fan out (event, payload) to many sinks without surfacing errors to caller.

    Concurrency model:
      - For each sink we respect a per-sink semaphore.
      - All sink tasks are awaited together per emit.
      - Caller never sees exceptions; optionally routed to on_error.

    Hooks:
      - on_before(event, payload): called once before dispatch
      - on_after(event, payload):  called once after dispatch
      - on_error(event, payload, exc, sink_name): error callback per failing sink
    """

    def __init__(
        self,
        *sinks: SinkFn,
        on_error: Optional[ErrorFn] = None,
        on_before: Optional[HookFn] = None,
        on_after: Optional[HookFn] = None,
        default_max_concurrency: int = 64,
        default_only: Optional[Iterable[str]] = None,
        default_skip: Optional[Iterable[str]] = None,
    ):
        self._sinks: List[SinkSpec] = []
        self._on_error = on_error
        self._on_before = on_before
        self._on_after = on_after
        self._default_max_concurrency = max(1, int(default_max_concurrency))
        self._default_only = set(default_only or [])
        self._default_skip = set(default_skip or [])

        for i, s in enumerate(sinks):
            self.add_sink(s, name=f"sink_{i}")

    # ------------------ public API ------------------

    def add_sink(
        self,
        fn: SinkFn,
        *,
        name: str = "sink",
        only: Optional[Iterable[str]] = None,
        skip: Optional[Iterable[str]] = None,
        filter_fn: Optional[FilterFn] = None,
        max_concurrency: Optional[int] = None,
    ) -> None:
        """Register a sink with optional filtering and concurrency bound."""
        spec = SinkSpec(
            fn=fn,
            name=name,
            only=set(only) if only is not None else (self._default_only or None),
            skip=set(skip) if skip is not None else (self._default_skip or None),
            filter_fn=filter_fn,
            max_concurrency=max_concurrency or self._default_max_concurrency,
        )
        self._sinks.append(spec)

    def remove_sink(self, name: str) -> bool:
        """Unregister sink by name; returns True if removed."""
        for i, s in enumerate(self._sinks):
            if s.name == name:
                del self._sinks[i]
                return True
        return False

    def list_sinks(self) -> List[str]:
        return [s.name for s in self._sinks]

    @asynccontextmanager
    async def temporary_sink(
        self,
        fn: SinkFn,
        *,
        name: str = "temp_sink",
        only: Optional[Iterable[str]] = None,
        skip: Optional[Iterable[str]] = None,
        filter_fn: Optional[FilterFn] = None,
        max_concurrency: Optional[int] = None,
    ):
        """Context-managed sink (auto-removed on exit)."""
        self.add_sink(
            fn, name=name, only=only, skip=skip, filter_fn=filter_fn, max_concurrency=max_concurrency
        )
        try:
            yield
        finally:
            self.remove_sink(name)

    async def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        """
        Dispatch to all eligible sinks.
        Never raises to caller; errors go to on_error (if provided) and are logged.
        """
        if self._on_before:
            try:
                self._on_before(event, payload)
            except Exception:
                # hook errors are ignored by design
                pass

        tasks: List[asyncio.Task] = []
        for spec in self._sinks:
            if not spec.allows(event, payload):
                continue
            tasks.append(asyncio.create_task(self._run_one(spec, event, payload)))

        if tasks:
            # Await all, but collapse exceptions into error handler
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, BaseException):
                    # The exception is already reported by _run_one; just continue
                    pass

        if self._on_after:
            try:
                self._on_after(event, payload)
            except Exception:
                pass

    # ------------------ internals ------------------

    async def _run_one(self, spec: SinkSpec, event: str, payload: Dict[str, Any]) -> None:
        # Respect per-sink concurrency
        async with spec._sem:
            try:
                ret = spec.fn(event, payload)
                if inspect.isawaitable(ret):
                    await ret
            except Exception as exc:
                # route to error hook if provided, otherwise log
                if self._on_error:
                    try:
                        self._on_error(event, payload, exc, spec.name)
                    except Exception:
                        # avoid recursive failures
                        pass
                else:
                    logger.warning(
                        "EmitBroadcaster sink '%s' failed for event '%s': %r",
                        spec.name, event, exc,
                        exc_info=True,
                    )


# ---------- Convenience sinks (optional, handy for quick wiring) ----------

async def noop_sink(_: str, __: Dict[str, Any]) -> None:
    """Does nothing; useful for testing."""
    return None

def console_sink(event: str, payload: Dict[str, Any]) -> None:
    """Simple sync sink that prints to stdout via logging."""
    logging.getLogger("emit.console").info("event=%s payload=%s", event, payload)

def make_prefix_filter(prefix: str) -> FilterFn:
    """Return a predicate that allows only events starting with prefix."""
    def _pred(evt: str, _: Dict[str, Any]) -> bool:
        return evt.startswith(prefix)
    return _pred
