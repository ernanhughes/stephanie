"""Unit tests for the Worker free-model bridge.

Cover the deterministic selection logic with a stub HTTP transport (no network,
no Worker required). A live smoke check lives in
``scripts/worker_free_model_check.py``.
"""

from __future__ import annotations

from stephanie.services.worker_model_bridge import (
    FreeRoute,
    WorkerModelBridge,
    WorkerModelProvider,
    _extract_output_text,
    _probe_succeeded,
    prefer_free_route,
)


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _StubSession:
    def __init__(self, *, discovered=None, probe_results=None, get_error=None):
        self._discovered = discovered or []
        self._probe_results = probe_results or {}
        self._get_error = get_error
        self.probe_calls: list[str] = []

    def get(self, url, params=None, timeout=None):
        if self._get_error:
            raise self._get_error
        return _Resp({"schema_version": "worker.cli-models.v1", "models": self._discovered})

    def post(self, url, json=None, timeout=None):
        model = (json or {}).get("model")
        self.probe_calls.append(model)
        return _Resp(self._probe_results.get(model, {"results": []}))


def _free_row(model_id: str) -> dict:
    provider, _, model = model_id.partition("/")
    return {
        "id": model_id,
        "provider": provider,
        "model": model,
        "free": True,
        "price_usd_per_million": 0,
    }


def _probe_ok(route_id: str) -> dict:
    return {
        "results": [
            {
                "model": route_id,
                "observation": {"status": "OK"},
                "provider_state": {"request_admissible": "ALLOWED"},
            }
        ]
    }


def _probe_blocked(route_id: str) -> dict:
    return {
        "results": [
            {
                "model": route_id,
                "observation": {"status": "FAILED", "failure": {"http_status": 503}},
                "provider_state": {"request_admissible": "BLOCKED", "status": "UNAVAILABLE"},
            }
        ]
    }


def test_free_route_parses_cli_row_and_defaults_to_training_allowed():
    route = FreeRoute.from_cli_row(_free_row("opencode/space-bunny-free"))
    assert route.id == "opencode/space-bunny-free"
    assert route.provider == "opencode"
    assert route.model == "space-bunny-free"
    assert route.price_usd_per_million == 0.0
    assert route.is_training_allowed


def test_probe_succeeded_requires_ok_and_not_blocked():
    assert _probe_succeeded(_probe_ok("a/b")) is True
    assert _probe_succeeded(_probe_blocked("a/b")) is False
    assert _probe_succeeded({"results": []}) is False
    assert _probe_succeeded({}) is False
    # OK status but explicitly blocked must not count as success.
    assert _probe_succeeded(
        {"results": [{"observation": {"status": "OK"}, "provider_state": {"request_admissible": "BLOCKED"}}]}
    ) is False


def test_select_free_route_returns_none_and_negative_caches_when_all_blocked():
    session = _StubSession(
        discovered=[_free_row("opencode/space-bunny-free"), _free_row("opencode/mimo-v2.6-flash-free")],
        probe_results={
            "opencode/space-bunny-free": _probe_blocked("opencode/space-bunny-free"),
            "opencode/mimo-v2.6-flash-free": _probe_blocked("opencode/mimo-v2.6-flash-free"),
        },
    )
    bridge = WorkerModelBridge(session=session)

    assert bridge.select_free_route() is None
    assert session.probe_calls == ["opencode/space-bunny-free", "opencode/mimo-v2.6-flash-free"]
    # Negative cache: no further probes within the TTL.
    assert bridge.select_free_route() is None
    assert len(session.probe_calls) == 2


def test_select_free_route_picks_the_first_admissible_route():
    session = _StubSession(
        discovered=[_free_row("opencode/space-bunny-free"), _free_row("opencode/mimo-v2.6-flash-free")],
        probe_results={
            "opencode/space-bunny-free": _probe_blocked("opencode/space-bunny-free"),
            "opencode/mimo-v2.6-flash-free": _probe_ok("opencode/mimo-v2.6-flash-free"),
        },
    )
    bridge = WorkerModelBridge(session=session)

    route = bridge.select_free_route()
    assert route is not None and route.id == "opencode/mimo-v2.6-flash-free"
    # Cached: a second call returns it without probing again.
    assert bridge.select_free_route().id == "opencode/mimo-v2.6-flash-free"
    assert len(session.probe_calls) == 2


def test_select_free_route_is_silent_when_worker_is_unreachable():
    bridge = WorkerModelBridge(session=_StubSession(get_error=ConnectionError("worker down")))
    assert bridge.select_free_route() is None
    assert "worker down" in bridge.status()["last_error"]


def test_prefer_free_route_never_raises_when_worker_is_absent():
    bridge = WorkerModelBridge(session=_StubSession(get_error=ConnectionError("no worker")))
    assert prefer_free_route(bridge=bridge) is None


def test_extract_output_text_handles_common_result_shapes():
    assert _extract_output_text({"output_text": "hello"}) == "hello"
    assert _extract_output_text({"result": {"output": "nested"}}) == "nested"
    assert _extract_output_text({"attempts": [{"output_text": "attempt"}]}) == "attempt"
    assert _extract_output_text({"unrelated": 1}) == ""
    assert _extract_output_text({}) == ""


def test_worker_model_provider_raises_when_no_free_route_is_admissible():
    import asyncio

    from stephanie.models.exceptions import ProviderInvocationError
    from stephanie.models.request import ModelRequest

    bridge = WorkerModelBridge(session=_StubSession(get_error=ConnectionError("no worker")))
    provider = WorkerModelProvider(bridge)

    try:
        asyncio.run(provider.invoke(ModelRequest(model="worker/free-route", prompt="hi")))
    except ProviderInvocationError as exc:
        assert "free" in str(exc).lower()
    else:
        raise AssertionError("Expected ProviderInvocationError when no route is admissible")
