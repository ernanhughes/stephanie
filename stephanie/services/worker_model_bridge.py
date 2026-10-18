"""Bridge to the local Worker's model runtime (free OpenCode routes).

Policy (agreed with the operator): **prefer a free OpenCode route when one is
admissible, and fall back to Stephanie's normal provider (Ollama) otherwise.**

Worker (``C:\\Projects\\worker``) is a loopback-only, unauthenticated service.
It exposes model *discovery* and *probing* over HTTP, but it has **no standalone
completion endpoint**: models are invoked inside a Work task through its
``CALL_MODEL`` capability. This bridge therefore:

* discovers free routes via ``GET /v1/models/cli?free=true``
* verifies one is actually callable via ``POST /v1/models/probe``
  (free routes have historically been denied through this API path, so probing
  is mandatory before routing anything to them)
* submits work via ``POST /v1/work`` when a route is admissible
* exposes a :class:`~stephanie.models.provider.ModelProvider` for the model
  runtime, and raises :class:`FreeRouteUnavailable` so callers fall back

Nothing here is invoked at import time, and the bridge is inert until a free
route is admissible.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from stephanie.models.exceptions import ProviderInvocationError
from stephanie.models.model import Model
from stephanie.models.provider import ModelProvider, ProviderHealth
from stephanie.models.request import ModelRequest
from stephanie.models.response import ModelResponse
from stephanie.models.usage import ModelUsage

log = logging.getLogger(__name__)

DEFAULT_WORKER_BASE_URL = "http://127.0.0.1:7317"
FREE_ROUTE_DATA_POLICY = "training_allowed"
_ADMISSIBLE_STATUSES = {"OK", "PASSED", "SUCCESS", "SUCCEEDED"}


class FreeRouteUnavailable(RuntimeError):
    """Raised when no free route is currently admissible (caller should fall back)."""


@dataclass(frozen=True)
class FreeRoute:
    """A free model route advertised by Worker's OpenCode model list."""

    id: str
    provider: str = ""
    model: str = ""
    price_usd_per_million: float = 0.0
    data_policy: str = FREE_ROUTE_DATA_POLICY

    @classmethod
    def from_cli_row(cls, row: dict) -> "FreeRoute":
        return cls(
            id=str(row.get("id") or ""),
            provider=str(row.get("provider") or ""),
            model=str(row.get("model") or ""),
            price_usd_per_million=float(row.get("price_usd_per_million") or 0.0),
        )

    @property
    def is_training_allowed(self) -> bool:
        return self.data_policy == FREE_ROUTE_DATA_POLICY


def _probe_succeeded(result: dict) -> bool:
    """True only when Worker reports the route was probed and is admissible."""
    rows = (result or {}).get("results") or []
    if not rows or not isinstance(rows[0], dict):
        return False
    first = rows[0]
    observation = first.get("observation") or {}
    state = first.get("provider_state") or {}
    status = str(observation.get("status") or "").upper()
    admissible = state.get("request_admissible")
    return status in _ADMISSIBLE_STATUSES and admissible not in {"BLOCKED", "DENIED"}


class WorkerModelBridge:
    """Discover, probe and invoke free models through the loopback Worker."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_WORKER_BASE_URL,
        timeout: float = 10.0,
        discovery_timeout: float = 90.0,
        probe_timeout: float = 200.0,
        cache_ttl_seconds: float = 300.0,
        session: Any = None,
        logger: logging.Logger | None = None,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        # Model discovery shells out to the OpenCode CLI on first use and can
        # take much longer than a normal request.
        self.discovery_timeout = float(discovery_timeout)
        self.probe_timeout = float(probe_timeout)
        self.cache_ttl_seconds = float(cache_ttl_seconds)
        self._session = session
        self._log = logger or log
        self._candidate: Optional[FreeRoute] = None
        self._candidate_at: float = 0.0
        self._blocked_until: float = 0.0
        self._last_error: str = ""

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def _http(self):
        if self._session is None:
            import requests  # project dependency; imported lazily so import stays light

            self._session = requests.Session()
        return self._session

    # ------------------------------------------------------------------
    # discovery / probing
    # ------------------------------------------------------------------
    def health(self) -> dict:
        response = self._http().get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def discover_free_models(self, *, refresh: bool = False) -> list[FreeRoute]:
        params: dict[str, str] = {"free": "true"}
        if refresh:
            params["refresh"] = "true"
        response = self._http().get(
            f"{self.base_url}/v1/models/cli", params=params, timeout=self.discovery_timeout
        )
        response.raise_for_status()
        rows = (response.json() or {}).get("models") or []
        routes = [
            FreeRoute.from_cli_row(row)
            for row in rows
            if isinstance(row, dict) and row.get("free") is True and row.get("id")
        ]
        self._log.info("Worker advertised %d free model route(s)", len(routes))
        return routes

    def probe(self, route_id: str, *, max_requests: int = 1) -> dict:
        response = self._http().post(
            f"{self.base_url}/v1/models/probe",
            json={"model": route_id, "max_requests": max(1, min(16, int(max_requests)))},
            timeout=self.probe_timeout,
        )
        response.raise_for_status()
        return response.json()

    def select_free_route(
        self,
        *,
        candidates: Optional[Iterable[FreeRoute]] = None,
        force: bool = False,
    ) -> Optional[FreeRoute]:
        """Return the first free route that probes as admissible, else ``None``.

        Falls back silently (returns ``None``) on any Worker/network problem, and
        negative-caches for ``cache_ttl_seconds`` so every call does not re-probe.
        """
        now = time.time()
        if not force:
            if now < self._blocked_until:
                return None
            if self._candidate and (now - self._candidate_at) < self.cache_ttl_seconds:
                return self._candidate

        try:
            routes = list(candidates) if candidates is not None else self.discover_free_models()
        except Exception as exc:  # network down, worker not running, etc.
            self._last_error = str(exc)
            self._log.warning("Worker model discovery unavailable: %s", exc)
            self._blocked_until = now + self.cache_ttl_seconds
            return None

        for route in routes:
            try:
                result = self.probe(route.id)
            except Exception as exc:
                self._last_error = str(exc)
                self._log.warning("Probe failed for free route %s: %s", route.id, exc)
                continue
            if _probe_succeeded(result):
                self._candidate = route
                self._candidate_at = now
                self._log.info("Selected admissible free route %s", route.id)
                return route
            self._last_error = f"not admissible: {route.id}"
            self._log.info("Free route %s is not admissible; skipping", route.id)

        self._blocked_until = now + self.cache_ttl_seconds
        self._log.warning(
            "No admissible free route (%d tried); falling back to the normal provider",
            len(routes),
        )
        return None

    # ------------------------------------------------------------------
    # invocation (through Worker's Work contract)
    # ------------------------------------------------------------------
    def submit_prompt(self, prompt: str, *, route: FreeRoute, app: str = "stephanie") -> str:
        """Run a prompt through Worker for ``route`` and return the result text.

        Worker invokes models inside a Work task, so this posts a Work request
        whose objective is the prompt and whose execution selects the route.
        Raises :class:`FreeRouteUnavailable` if Worker reports no usable output.
        """
        body = {
            "schema": "worker.work-request.v1",
            "app": app,
            "objective": prompt,
            "execution": {"model": route.id},
            "options": {"acceptance": "automatic"},
        }
        response = self._http().post(f"{self.base_url}/v1/work", json=body, timeout=self.probe_timeout)
        response.raise_for_status()
        payload = response.json() or {}
        text = _extract_output_text(payload)
        if not text:
            raise FreeRouteUnavailable(
                f"Worker returned no output text for route {route.id}"
            )
        return text

    # ------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "base_url": self.base_url,
            "candidate": self._candidate.id if self._candidate else None,
            "blocked_until": self._blocked_until or None,
            "last_error": self._last_error or None,
        }


def _extract_output_text(payload: dict) -> str:
    """Best-effort extraction of generated text from a worker.work-result payload."""
    if not isinstance(payload, dict):
        return ""
    for key in ("output_text", "output", "text", "stdout", "answer"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ("output_text", "output", "text", "summary"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for attempt in (payload.get("attempts") or []):
        if isinstance(attempt, dict):
            value = attempt.get("output_text") or attempt.get("output")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


class WorkerModelProvider(ModelProvider):
    """Model-runtime provider that routes through the local Worker when free is available."""

    def __init__(self, bridge: WorkerModelBridge, *, provider_name: str = "worker"):
        self.bridge = bridge
        self._provider_name = provider_name

    def supports(self, model: Model) -> bool:
        return model.provider == self._provider_name

    async def health(self) -> ProviderHealth:
        try:
            info = self.bridge.health()
            return ProviderHealth(healthy=True, provider=self._provider_name, detail=str(info.get("status", "")))
        except Exception as exc:  # pragma: no cover - defensive
            return ProviderHealth(healthy=False, provider=self._provider_name, detail=str(exc))

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        model = request.model if isinstance(request.model, Model) else Model.from_ref(str(request.model))
        route = self.bridge.select_free_route()
        if route is None:
            raise ProviderInvocationError(
                self._provider_name, model.id, "no admissible free OpenCode route"
            )
        prompt = request.prompt or next(
            (m.content for m in reversed(request.messages or []) if m.role == "user"), ""
        )
        started = time.perf_counter()
        try:
            text = await asyncio.to_thread(self.bridge.submit_prompt, prompt, route=route)
        except FreeRouteUnavailable as exc:
            raise ProviderInvocationError(self._provider_name, model.id, str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ModelResponse(
            request_id=request.trace_id or "",
            model_id=route.id,
            provider=self._provider_name,
            output_text=text,
            usage=ModelUsage(latency_ms=latency_ms),
            latency_ms=latency_ms,
            finish_reason="stop",
            created_at=datetime.now(timezone.utc),
            metadata={"free_route": route.id, "data_policy": route.data_policy},
        )


def prefer_free_route(
    *,
    base_url: str = DEFAULT_WORKER_BASE_URL,
    bridge: Optional[WorkerModelBridge] = None,
    logger: logging.Logger | None = None,
) -> Optional[FreeRoute]:
    """Convenience: return an admissible free route, or ``None`` to fall back.

    Never raises for the "no free route" case — callers treat ``None`` as
    "use the normal provider".
    """
    bridge = bridge or WorkerModelBridge(base_url=base_url, logger=logger)
    try:
        return bridge.select_free_route()
    except Exception as exc:  # pragma: no cover - defensive
        (logger or log).warning("Free-route selection failed: %s", exc)
        return None
