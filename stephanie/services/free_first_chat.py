"""Small, deterministic OpenCode Free-first policy for SIS chat.

This is conversation routing policy, not a model catalogue. Route eligibility
is explicitly configured by the operator and checked against the message data
class before any request leaves Stephanie.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import email.utils
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class RouteFailure:
    route: str
    state: str
    detail: str
    retry_after_seconds: float | None = None


class FreeFirstChatRouter:
    """Call configured eligible free routes once each, then return a result.

    Unknown and unsupported failures stop routing. The caller may use its
    separately configured local route for known availability/quota failures.
    Cooldowns are process-local and never represented as provider quota data.
    """

    _FALLTHROUGH = {"rate_limited", "free_allocation_exhausted", "provider_unavailable", "model_unavailable"}

    def __init__(self, config: dict[str, Any], *, api_key: str | None):
        self.config = dict(config or {})
        self.api_key = api_key or ""
        self._cooldowns: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled"))

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        call: Callable[..., Awaitable[Any]],
        data_class: str = "personal",
        required_capabilities: set[str] | None = None,
        pinned: bool = False,
        pinned_route: str | None = None,
        session_id: str | None = None,
        timeout: int = 90,
        max_tokens: int = 768,
    ) -> tuple[Any | None, dict[str, Any]]:
        if not self.enabled:
            return None, {"state": "disabled"}
        if pinned:
            return None, {"state": "explicitly_pinned"}
        if not self.api_key:
            return None, {
                "state": "authentication_failed", "fallback_allowed": bool(self.config.get("allow_local_fallback", True)),
                "failures": [{"route": "opencode", "state": "authentication_failed", "detail": "OPENCODE_API_KEY is not configured"}],
            }
        privacy_eligible = [
            route for route in self.config.get("routes", [])
            if route.get("economic_tier") in {"free", "subscription_included"}
            and data_class in route.get("allowed_data_classes", [])
            and (
                (route.get("provider") == "opencode" and route.get("endpoint") == "https://opencode.ai/inference/openai/v1")
                or (route.get("provider") == "opencode-go" and route.get("endpoint") == "https://opencode.ai/inference/go/openai/v1")
            )
        ]
        if not privacy_eligible:
            return None, {"state": "no_privacy_eligible_free_route"}
        eligible = [route for route in privacy_eligible if
                    (required_capabilities or set()).issubset(set(route.get("capabilities", [])))]
        if not eligible:
            return None, {"state": "unsupported_capability", "fallback_allowed": False,
                          "required_capabilities": sorted(required_capabilities or set())}
        if pinned_route:
            requested = pinned_route.removeprefix("opencode/")
            eligible = [route for route in eligible if route.get("model") == requested]
            if not eligible:
                return None, {"state": "unsupported_pinned_route", "fallback_allowed": False,
                              "pinned_route": pinned_route}
        failures: list[RouteFailure] = []
        for route in eligible:
            route_id = str(route["model"])
            lock = self._locks.setdefault(route_id, asyncio.Lock())
            async with lock:
                now = time.monotonic()
                if self._cooldowns.get(route_id, 0) > now:
                    failures.append(RouteFailure(route_id, "rate_limited", "route cooldown active"))
                    continue
                try:
                    request_args = {
                        "model": f"openai/{route_id}",
                        "api_base": route.get("endpoint") or self.config.get("endpoint"),
                        "api_key": self.api_key,
                        "messages": messages,
                        "timeout": timeout,
                        "max_tokens": max_tokens,
                    }
                    if route.get("provider") == "opencode-go":
                        if not session_id:
                            return None, {"state": "session_id_required", "fallback_allowed": False}
                        request_args["extra_headers"] = {"x-opencode-session": session_id}
                    response = await call(**request_args)
                    provider = str(route.get("provider", "opencode"))
                    return response, {
                        "state": "selected", "route": f"{provider}/{route_id}",
                        "reason": "first eligible configured route",
                        "economic_tier": route.get("economic_tier"), "data_class": data_class,
                        "capabilities": sorted(required_capabilities or set()),
                        "pinned": bool(pinned_route),
                    }
                except Exception as exc:
                    failure = self._classify(route_id, exc)
                    failures.append(failure)
                    if pinned_route:
                        return None, {"state": failure.state, "fallback_allowed": False,
                                      "pinned": True, "failures": [f.__dict__ for f in failures]}
                    if failure.state in {"rate_limited", "free_allocation_exhausted"}:
                        cooldown = failure.retry_after_seconds
                        if cooldown is None:
                            cooldown = float(self.config.get("default_rate_limit_cooldown_seconds", 60))
                        self._cooldowns[route_id] = time.monotonic() + max(1.0, cooldown)
                    if failure.state not in self._FALLTHROUGH:
                        return None, self._failure_summary(failure, failures)
        return None, {
            "state": "free_routes_unavailable", "fallback_allowed": bool(self.config.get("allow_local_fallback", True)),
            "failures": [f.__dict__ for f in failures],
        }

    @staticmethod
    def _classify(route: str, exc: Exception) -> RouteFailure:
        response = getattr(exc, "response", None)
        status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
        text = str(exc).lower()
        headers = getattr(response, "headers", {}) or {}
        retry_after = None
        reset_header = None
        reset_value = None
        for header in ("retry-after", "ratelimit-reset", "x-ratelimit-reset"):
            value = headers.get(header)
            if value is not None:
                reset_header, reset_value = header, value
                break
        if reset_value is not None:
            try:
                retry_after = float(reset_value)
                if reset_header == "x-ratelimit-reset" and retry_after > 1_000_000_000:
                    retry_after = max(0.0, retry_after - time.time())
            except (TypeError, ValueError):
                try:
                    reset_at = email.utils.parsedate_to_datetime(str(reset_value))
                    if reset_at.tzinfo is None:
                        reset_at = reset_at.replace(tzinfo=dt.timezone.utc)
                    retry_after = max(0.0, (reset_at - dt.datetime.now(dt.timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        if status in (401, 403) or any(s in text for s in ("unauthorized", "authentication", "permission denied", "forbidden")):
            state = "authentication_failed" if status == 401 or "unauthor" in text or "authentication" in text else "permission_denied"
        elif status == 429 or "rate limit" in text or "too many requests" in text:
            state = "free_allocation_exhausted" if any(s in text for s in ("quota", "allocation", "usage limit", "credits exhausted")) else "rate_limited"
        elif status == 404 or "model not found" in text or "model unavailable" in text:
            state = "model_unavailable"
        elif status == 400 and any(s in text for s in ("unsupported", "capability", "not supported")):
            state = "unsupported_capability"
        elif (isinstance(status, int) and status >= 500) or any(s in text for s in ("connection error", "connection refused", "provider unavailable", "timed out")):
            state = "provider_unavailable"
        else:
            state = "unknown"
        # Provider exceptions can echo request headers in some SDKs. Retain
        # only type/state metadata so credentials and prompt fragments cannot
        # leak into persisted ChatTurn diagnostics.
        return RouteFailure(route, state, type(exc).__name__, retry_after)

    @staticmethod
    def _failure_summary(failure: RouteFailure, failures: list[RouteFailure]) -> dict[str, Any]:
        return {"state": failure.state, "fallback_allowed": failure.state in FreeFirstChatRouter._FALLTHROUGH,
                "failures": [f.__dict__ for f in failures]}
