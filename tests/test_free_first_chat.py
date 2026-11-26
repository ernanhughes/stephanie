from __future__ import annotations

import asyncio
from types import SimpleNamespace
import pytest

from stephanie.services.llm_service import LLMService
from stephanie.services.free_first_chat import FreeFirstChatRouter

pytestmark = pytest.mark.long


ROUTES = {
    "enabled": True,
    "allow_local_fallback": True,
    "default_rate_limit_cooldown_seconds": 60,
    "routes": [
        {"model": "deepseek-v4.1-flash-free", "provider": "opencode", "economic_tier": "free",
         "endpoint": "https://opencode.ai/inference/openai/v1", "allowed_data_classes": ["personal"],
         "capabilities": ["conversation"]},
        {"model": "hy4-flash-free", "provider": "opencode", "economic_tier": "free",
         "endpoint": "https://opencode.ai/inference/openai/v1", "allowed_data_classes": ["personal"],
         "capabilities": ["conversation"]},
    ],
}


class ApiFailure(Exception):
    def __init__(self, status_code, *, retry_after=None, reset=None, detail=None):
        super().__init__(detail or f"HTTP {status_code}")
        headers = {}
        if retry_after is not None:
            headers["retry-after"] = str(retry_after)
        if reset is not None:
            headers["ratelimit-reset"] = str(reset)
        self.response = SimpleNamespace(
            status_code=status_code,
            headers=headers,
        )


def test_preferred_free_route_is_selected_and_reports_identity():
    calls = []

    async def call(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    response, route = asyncio.run(FreeFirstChatRouter(ROUTES, api_key="secret").complete(
        [{"role": "user", "content": "hello"}], call=call))

    assert response == {"ok": True}
    assert route["route"] == "opencode/deepseek-v4.1-flash-free"
    assert route["economic_tier"] == "free"
    assert calls[0]["model"] == "openai/deepseek-v4.1-flash-free"
    assert calls[0]["messages"][0]["content"] == "hello"
    assert len(calls) == 1


def test_quota_failure_uses_one_alternative_without_losing_history():
    calls = []

    async def call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ApiFailure(429, retry_after=45)
        return {"ok": True}

    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "answer"},
               {"role": "user", "content": "new"}]
    response, route = asyncio.run(FreeFirstChatRouter(ROUTES, api_key="secret").complete(history, call=call))
    assert response == {"ok": True}
    assert route["route"] == "opencode/hy4-flash-free"
    assert len(calls) == 2
    assert calls[1]["messages"] == history


def test_auth_failure_stops_and_does_not_rotate_free_routes():
    calls = []

    async def call(**kwargs):
        calls.append(kwargs)
        raise ApiFailure(403)

    response, route = asyncio.run(FreeFirstChatRouter(ROUTES, api_key="secret").complete(
        [{"role": "user", "content": "synthetic"}], call=call))
    assert response is None
    assert route["state"] == "permission_denied"
    assert route["fallback_allowed"] is False
    assert len(calls) == 1


def test_unknown_error_fails_closed_and_privacy_ineligible_data_is_not_sent():
    calls = []

    async def call(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("unclassified failure")

    router = FreeFirstChatRouter(ROUTES, api_key="secret")
    _, unknown = asyncio.run(router.complete([{"role": "user", "content": "x"}], call=call))
    assert unknown["state"] == "unknown"
    assert not unknown["fallback_allowed"]

    _, private = asyncio.run(router.complete(
        [{"role": "user", "content": "source"}], call=call, data_class="confidential_source"))
    assert private["state"] == "no_privacy_eligible_free_route"
    assert len(calls) == 1


def test_explicit_pin_and_missing_credentials_do_not_call_provider():
    calls = []

    async def call(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    router = FreeFirstChatRouter(ROUTES, api_key="")
    _, pinned = asyncio.run(router.complete([{"role": "user", "content": "x"}], call=call, pinned=True))
    _, missing_key = asyncio.run(router.complete([{"role": "user", "content": "x"}], call=call))
    assert pinned["state"] == "explicitly_pinned"
    assert missing_key["state"] == "authentication_failed"
    assert missing_key["fallback_allowed"]
    assert calls == []


def test_owner_pinned_route_is_selected_alone_and_never_rotated():
    calls = []

    async def call(**kwargs):
        calls.append(kwargs["model"])
        raise ApiFailure(429)

    router = FreeFirstChatRouter(ROUTES, api_key="secret")
    _, result = asyncio.run(router.complete(
        [{"role": "user", "content": "x"}], call=call, pinned_route="opencode/hy4-flash-free"))
    assert result["state"] == "rate_limited"
    assert result["pinned"] is True
    assert result["fallback_allowed"] is False
    assert calls == ["openai/hy4-flash-free"]


def test_capability_gate_and_cooldown_prevent_repeating_a_limited_route():
    calls = []

    async def call(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "openai/deepseek-v4.1-flash-free":
            raise ApiFailure(429, retry_after=90)
        return {"ok": True}

    router = FreeFirstChatRouter(ROUTES, api_key="secret")
    _, unsupported = asyncio.run(router.complete(
        [{"role": "user", "content": "x"}], call=call, required_capabilities={"coding"}))
    assert unsupported["state"] == "unsupported_capability"
    assert calls == []

    _, first = asyncio.run(router.complete([{"role": "user", "content": "x"}], call=call))
    assert first["route"] == "opencode/hy4-flash-free"
    first_call_count = calls.count("openai/deepseek-v4.1-flash-free")
    _, second = asyncio.run(router.complete([{"role": "user", "content": "next"}], call=call))
    assert second["route"] == "opencode/hy4-flash-free"
    assert calls.count("openai/deepseek-v4.1-flash-free") == first_call_count


def test_exhausted_free_allocation_also_enters_cooldown():
    calls = []

    async def call(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "openai/deepseek-v4.1-flash-free":
            raise ApiFailure(429, detail="free quota exhausted")
        return {"ok": True}

    router = FreeFirstChatRouter(ROUTES, api_key="secret")
    _, first = asyncio.run(router.complete([{"role": "user", "content": "x"}], call=call))
    assert first["route"] == "opencode/hy4-flash-free"
    initial = calls.count("openai/deepseek-v4.1-flash-free")
    _, second = asyncio.run(router.complete([{"role": "user", "content": "y"}], call=call))
    assert second["route"] == "opencode/hy4-flash-free"
    assert calls.count("openai/deepseek-v4.1-flash-free") == initial


def test_nonfree_routes_are_never_selected_and_failure_classes_are_explicit():
    calls = []
    routes = {**ROUTES, "routes": [*ROUTES["routes"], {
        "model": "gpt-6-luna", "provider": "opencode", "economic_tier": "subscription",
        "endpoint": "https://opencode.ai/inference/openai/v1", "allowed_data_classes": ["personal"],
        "capabilities": ["conversation"],
    }]}

    async def call(**kwargs):
        calls.append(kwargs["model"])
        raise ApiFailure(503)

    result, state = asyncio.run(FreeFirstChatRouter(routes, api_key="secret").complete(
        [{"role": "user", "content": "x"}], call=call))
    assert result is None
    assert state["state"] == "free_routes_unavailable"
    assert calls == ["openai/deepseek-v4.1-flash-free", "openai/hy4-flash-free"]


def test_go_route_requires_and_sends_stable_opencode_session_header():
    calls = []
    go_config = {
        "enabled": True,
        "routes": [{
            "model": "deepseek-v4.1-flash",
            "provider": "opencode-go",
            "economic_tier": "subscription_included",
            "endpoint": "https://opencode.ai/inference/go/openai/v1",
            "allowed_data_classes": ["approved_external"],
            "capabilities": ["conversation"],
        }],
    }

    async def call(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    router = FreeFirstChatRouter(go_config, api_key="synthetic-test-key")
    _, missing = asyncio.run(router.complete(
        [], call=call, data_class="approved_external", session_id=None
    ))
    assert missing["state"] == "session_id_required"
    assert not calls
    response, selected = asyncio.run(router.complete(
        [], call=call, data_class="approved_external", session_id="ses_stable_test"
    ))
    assert response == {"ok": True}
    assert selected["route"] == "opencode-go/deepseek-v4.1-flash"
    assert calls[0]["extra_headers"] == {"x-opencode-session": "ses_stable_test"}


def test_automatic_policy_does_not_use_subscription_route_when_disabled():
    calls = []
    go_config = {"enabled": True, "routes": [{
        "model": "deepseek-v4.1-flash", "provider": "opencode-go",
        "economic_tier": "subscription_included",
        "endpoint": "https://opencode.ai/inference/go/openai/v1",
        "allowed_data_classes": ["approved_external"], "capabilities": ["conversation"],
    }]}

    async def call(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    _, blocked = asyncio.run(FreeFirstChatRouter(go_config, api_key="synthetic-test-key").complete(
        [], call=call, data_class="approved_external", subscription_allowed=False,
        session_id="stable-session",
    ))
    assert blocked["state"] == "no_privacy_eligible_free_route"
    assert not calls


def test_explicit_route_is_selected_exactly_and_rejects_unapproved_data():
    calls = []
    go_config = {"enabled": True, "routes": [{
        "model": "deepseek-v4.1-flash", "provider": "opencode-go",
        "economic_tier": "subscription_included",
        "endpoint": "https://opencode.ai/inference/go/openai/v1",
        "allowed_data_classes": ["approved_external"], "capabilities": ["conversation"],
    }]}

    async def call(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    router = FreeFirstChatRouter(go_config, api_key="synthetic-test-key")
    _, blocked = asyncio.run(router.complete([], call=call, data_class="private_project",
        requested_route="opencode-go/deepseek-v4.1-flash", session_id="stable"))
    assert blocked["state"] == "not_approved_for_data"
    assert calls == []

    _, denied = asyncio.run(router.complete([], call=call, data_class="approved_external",
        requested_route="opencode-go/deepseek-v4.1-flash", subscription_allowed=False, session_id="stable"))
    assert denied["state"] == "subscription_route_not_authorized"
    assert calls == []

    response, selected = asyncio.run(router.complete([], call=call, data_class="approved_external",
        requested_route="opencode-go/deepseek-v4.1-flash", session_id="stable"))
    assert response == {"ok": True}
    assert selected["route"] == "opencode-go/deepseek-v4.1-flash"
    assert len(calls) == 1


def test_explicit_metered_route_is_rejected_even_when_configured():
    calls = []
    config = {"enabled": True, "routes": [{
        "model": "gpt-6-luna", "provider": "opencode",
        "economic_tier": "metered", "endpoint": "https://opencode.ai/inference/openai/v1",
        "allowed_data_classes": ["approved_external"], "capabilities": ["conversation"],
    }]}

    async def call(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    _, blocked = asyncio.run(FreeFirstChatRouter(config, api_key="synthetic-test-key").complete(
        [], call=call, data_class="approved_external", requested_route="opencode/gpt-6-luna",
    ))
    assert blocked["state"] == "metered_route_disabled"
    assert not calls


def test_explicit_route_with_changed_transport_is_rejected():
    calls = []
    config = {"enabled": True, "routes": [{
        "model": "deepseek-v4.1-flash", "provider": "opencode-go",
        "economic_tier": "subscription_included", "endpoint": "https://example.invalid/v1",
        "allowed_data_classes": ["approved_external"], "capabilities": ["conversation"],
    }]}

    async def call(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    _, blocked = asyncio.run(FreeFirstChatRouter(config, api_key="synthetic-test-key").complete(
        [], call=call, data_class="approved_external",
        requested_route="opencode-go/deepseek-v4.1-flash", session_id="stable-session",
    ))
    assert blocked["state"] == "requested_route_unavailable"
    assert blocked["reason"] == "unsupported_transport"
    assert not calls


def test_explicit_local_model_selection_overrides_default_without_external_routing(monkeypatch):
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"model": "qwen2", "choices": [{"message": {"content": "local answer"}, "finish_reason": "stop"}]}

    monkeypatch.setattr("stephanie.services.llm_service.litellm.acompletion", completion)

    class Logger:
        def log(self, *_args, **_kwargs): pass

    service = LLMService(cfg={"chat_runtime": {"routing": ROUTES}}, memory=SimpleNamespace(),
                         container=SimpleNamespace(), logger=Logger())
    result = asyncio.run(service.achat(
        [{"role": "user", "content": "private project question"}],
        context={"conversation_id": 4, "data_class": "private_project", "sis_data_policy_enforced": True,
                 "selected_model": "local/ollama/qwen2:latest"},
        agent_cfg={"model": {"name": "ollama/qwen3", "api_base": "http://localhost:11434"}},
        agent_name="sis_chat",
    ))
    assert result["text"] == "local answer"
    assert result["model_name"] == "qwen2"
    assert result["provider"] == "ollama"
    assert len(calls) == 1
    assert calls[0]["model"] == "ollama/qwen2:latest"
    assert calls[0]["api_base"] == "http://localhost:11434"


@pytest.mark.parametrize("model,reason", [
    ("local/ollama/gpt-oss:20b-cloud", "Cloud-tagged"),
    ("local/ollama/nomic-embed-text:latest", "Embedding-only"),
])
def test_local_cloud_and_embedding_routes_are_rejected_before_inference(model, reason):
    service = LLMService(cfg={}, memory=SimpleNamespace(), container=SimpleNamespace(),
                         logger=SimpleNamespace(log=lambda *_args, **_kwargs: None))

    with pytest.raises(ValueError, match=reason):
        asyncio.run(service.achat(
            [{"role": "user", "content": "synthetic"}],
            context={"conversation_id": 2, "selected_model": model, "sis_data_policy_enforced": True},
            agent_cfg={"model": {"name": "ollama/mistral-small:latest", "api_base": "http://127.0.0.1:11434"}},
            agent_name="sis_chat",
            add_to_history=False,
        ))


def test_automatic_private_fallback_rejects_cloud_tagged_default_before_inference(monkeypatch):
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "must not run"}}]}

    monkeypatch.setattr("stephanie.services.llm_service.litellm.acompletion", completion)
    service = LLMService(cfg={}, memory=SimpleNamespace(), container=SimpleNamespace(),
                         logger=SimpleNamespace(log=lambda *_args, **_kwargs: None))

    with pytest.raises(ValueError, match="Cloud-tagged Ollama models are not eligible as a local fallback"):
        asyncio.run(service.achat(
            [{"role": "user", "content": "synthetic private project question"}],
            context={"conversation_id": 3, "selected_model": "automatic", "data_class": "private_project",
                     "sis_data_policy_enforced": True},
            agent_cfg={"model": {"name": "ollama/gpt-oss:20b-cloud", "api_base": "http://127.0.0.1:11434"}},
            agent_name="sis_chat",
            add_to_history=False,
        ))
    assert not calls


def test_automatic_private_fallback_rejects_non_loopback_ollama_endpoint(monkeypatch):
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "must not run"}}]}

    monkeypatch.setattr("stephanie.services.llm_service.litellm.acompletion", completion)
    service = LLMService(cfg={}, memory=SimpleNamespace(), container=SimpleNamespace(),
                         logger=SimpleNamespace(log=lambda *_args, **_kwargs: None))

    with pytest.raises(ValueError, match="only a loopback Ollama fallback"):
        asyncio.run(service.achat(
            [{"role": "user", "content": "synthetic private question"}],
            context={"conversation_id": 4, "selected_model": "automatic", "data_class": "private_project",
                     "sis_data_policy_enforced": True},
            agent_cfg={"model": {"name": "ollama/mistral-small:latest", "api_base": "https://ollama.example.invalid/v1"}},
            agent_name="sis_chat",
            add_to_history=False,
        ))
    assert not calls


def test_sis_policy_rejects_external_model_override_before_provider_call(monkeypatch):
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "should not run"}}]}

    monkeypatch.setattr("stephanie.services.llm_service.litellm.acompletion", completion)

    class Logger:
        def log(self, *_args, **_kwargs): pass

    service = LLMService(
        cfg={"chat_runtime": {"routing": ROUTES}}, memory=SimpleNamespace(),
        container=SimpleNamespace(), logger=Logger(),
    )
    try:
        asyncio.run(service.achat(
            [{"role": "user", "content": "private"}],
            context={"conversation_id": 4, "data_class": "private_project", "sis_data_policy_enforced": True},
            agent_cfg={"model": {"name": "openai/other-model", "api_base": "https://provider.invalid/v1"}},
            agent_name="sis_chat",
        ))
    except ValueError as exc:
        assert "loopback Ollama fallback" in str(exc)
    else:
        raise AssertionError("external model override should be rejected")
    assert calls == []

    unsupported = FreeFirstChatRouter._classify("route", ApiFailure(400, detail="unsupported capability"))
    assert unsupported.state == "unsupported_capability"
    assert FreeFirstChatRouter._classify("route", ApiFailure(401)).state == "authentication_failed"
    assert FreeFirstChatRouter._classify("route", ApiFailure(404)).state == "model_unavailable"
    assert FreeFirstChatRouter._classify("route", ApiFailure(503)).state == "provider_unavailable"
    assert FreeFirstChatRouter._classify("route", ApiFailure(429, reset=30)).retry_after_seconds == 30


def test_all_free_routes_exhausted_enters_existing_local_chat_path(monkeypatch):
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        if kwargs["api_base"].startswith("https://opencode.ai"):
            raise ApiFailure(429)
        return {"model": "qwen3", "choices": [{"message": {"content": "local answer"}, "finish_reason": "stop"}]}

    monkeypatch.setattr("stephanie.services.llm_service.litellm.acompletion", completion)

    class Logger:
        def log(self, *_args, **_kwargs):
            pass

    service = LLMService(
        cfg={"chat_runtime": {"routing": ROUTES}}, memory=SimpleNamespace(),
        container=SimpleNamespace(), logger=Logger(),
    )
    messages = [{"role": "user", "content": "previous turn"},
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "new turn"}]
    result = asyncio.run(service.achat(
        messages, context={"conversation_id": 9, "data_class": "personal"},
        agent_cfg={"model": {"name": "ollama/qwen3", "api_base": "http://localhost:11434", "max_tokens": 77}},
        agent_name="sis_chat",
    ))

    assert result["text"] == "local answer"
    assert result["routing"]["state"] == "local_fallback"
    assert len(calls) == 3  # each free route once, then the preconfigured local route
    assert calls[-1]["api_base"] == "http://localhost:11434"
    assert calls[-1]["messages"] == messages
    assert calls[0]["model"] == "openai/deepseek-v4.1-flash-free"
    assert calls[1]["model"] == "openai/hy4-flash-free"


def test_llm_service_reports_upstream_route_and_compatibility_transport(monkeypatch):
    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)
        return {"model": "deepseek-v4.1-flash-free", "id": "response-1",
                "choices": [{"message": {"content": "free answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2}}

    monkeypatch.setattr("stephanie.services.llm_service.litellm.acompletion", completion)

    class Logger:
        def log(self, *_args, **_kwargs):
            pass

    config = {**ROUTES, "api_key": "test-key"}
    service = LLMService(
        cfg={"chat_runtime": {"routing": config}}, memory=SimpleNamespace(),
        container=SimpleNamespace(), logger=Logger(),
    )
    result = asyncio.run(service.achat(
        [{"role": "user", "content": "hello"}], context={"conversation_id": 4, "data_class": "personal"},
        agent_cfg={"model": {"name": "ollama/qwen3", "api_base": "http://localhost:11434"}},
        agent_name="sis_chat",
    ))
    assert result["model_name"] == "opencode/deepseek-v4.1-flash-free"
    assert result["provider"] == "opencode"
    assert result["provider_transport"] == "openai-compatible"
    assert captured["api_key"] == "test-key"
    assert "test-key" not in repr(result)
