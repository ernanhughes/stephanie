"""Stage 1 contract + compatibility tests (§17). stdlib-only, no I/O."""
from __future__ import annotations

import asyncio
from decimal import Decimal

from stephanie.models import (
    DefaultModelPolicy,
    LiteLLMProvider,
    Model,
    ModelPolicyRejected,
    ModelRegistry,
    ModelRequest,
    ModelResponse,
    ModelSpecAdapter,
    ModelUsage,
    NullPricingService,
    PolicyConstraints,
    RoutingContext,
    SimpleModelRouter,
    StaticPricingService,
    PriceEntry,
    StubProvider,
)
from stephanie.models.response import ModelResponse as MR
from stephanie.models.usage import ModelUsage as MU
from stephanie.services.model_runtime import InMemoryUsageRecorder, ModelRuntime
from stephanie.types.model import ModelSpec


def _run(coro):
    return asyncio.run(coro)


def test_model_from_ref_prefixes():
    assert Model.from_ref("opencode-go:muse-spark").provider == "opencode-go"
    assert Model.from_ref("opencode:foo").provider == "opencode"
    assert Model.from_ref("llamacpp:deepseek").provider == "llamacpp"
    assert Model.from_ref("qwen3.6:27b").provider == "ollama"  # bare default
    assert Model.from_ref("ollama:qwen3").local is True
    assert Model.from_ref("openai:gpt-5.6").local is False


def test_model_from_model_spec_compat():
    spec = ModelSpec(name="ollama/qwen:0.5b", api_base="http://localhost:11434")
    model = Model.from_model_spec(spec)
    assert model.provider == "ollama"
    assert "qwen" in model.name
    assert model.metadata["api_base"] == "http://localhost:11434"
    assert ModelSpecAdapter.to_model(spec).id == model.id


def test_registry_resolve_and_capability_filter():
    from stephanie.models import ModelCapabilities

    reg = ModelRegistry()
    reg.register_model(Model.from_ref("ollama:qwen3"))
    reg.register_model(
        Model.from_ref("openai:gpt-5.6", capabilities=ModelCapabilities(tool_use=True))
    )
    assert reg.resolve("ollama:qwen3").id == "ollama:qwen3"
    assert reg.resolve("opencode-go:new-model").provider == "opencode-go"  # ad-hoc
    assert [m.id for m in reg.available_models(capability="tool_use")] == ["openai:gpt-5.6"]


def test_routing_returns_list_and_preserves_task_type():
    reg = ModelRegistry()
    reg.register_model(Model.from_ref("ollama:qwen3"))
    router = SimpleModelRouter(reg)
    req = ModelRequest(model="ollama:qwen3", prompt="hi", task_type="book.argument.review")
    routed = router.route(req, RoutingContext.from_request(req))
    assert isinstance(routed, list) and len(routed) == 1
    assert RoutingContext.from_request(req).task_type == "book.argument.review"


def test_policy_local_only_and_critic_differs():
    policy = DefaultModelPolicy(PolicyConstraints(local_only=True))
    local = Model.from_ref("ollama:qwen3")
    remote = Model.from_ref("openai:gpt-5.6")
    decision = policy.evaluate(ModelRequest(model="x", prompt="hi"), [local, remote])
    assert decision.allowed and decision.preferred_models == [local.id]

    policy2 = DefaultModelPolicy(PolicyConstraints(generator_model_id=local.id))
    d2 = policy2.evaluate(ModelRequest(model="x", prompt="hi"), [local, remote])
    assert d2.preferred_models == [remote.id]

    d3 = policy2.evaluate(ModelRequest(model="x", prompt="hi"), [local])
    assert d3.allowed is False


def test_stub_provider_contract():
    provider = StubProvider()
    req = ModelRequest(model=Model.from_ref("stub:a"), prompt="hello", trace_id="t-1")
    resp = _run(provider.invoke(req))
    assert isinstance(resp, ModelResponse)
    assert resp.provider == "stub"
    assert resp.model_id == "stub:a"
    assert resp.output_text.endswith("hello")
    assert resp.latency_ms is not None
    assert resp.usage.input_tokens == 1
    assert resp.metadata["trace_id"] == "t-1"
    assert resp.created_at is not None


def test_runtime_invoke_records_usage_and_prices():
    reg = ModelRegistry()
    reg.register_model(Model.from_ref("stub:a"))
    reg.register_provider("stub", StubProvider())
    pricing = StaticPricingService(
        {"stub:a": PriceEntry(input_per_1k_usd=Decimal("1"), output_per_1k_usd=Decimal("2"))}
    )
    recorder = InMemoryUsageRecorder()
    runtime = ModelRuntime(registry=reg, pricing=pricing, usage_recorder=recorder)
    resp = _run(runtime.invoke(ModelRequest(model="stub:a", prompt="one two", task_type="generic.summarize")))
    assert resp.output_text.endswith("one two")
    assert resp.usage.estimated_cost_usd is not None
    assert len(recorder.records) == 1
    rec = recorder.records[0]
    assert rec.task_type == "generic.summarize" and rec.success is True


def test_runtime_unknown_pricing_stays_unknown():
    reg = ModelRegistry()
    reg.register_provider("stub", StubProvider())
    runtime = ModelRuntime(registry=reg, pricing=NullPricingService())
    resp = _run(runtime.invoke(ModelRequest(model="stub:a", prompt="hi")))
    assert resp.usage.estimated_cost_usd is None  # never silently zero


def test_runtime_policy_rejection_and_failure_recording():
    reg = ModelRegistry()
    only = Model.from_ref("stub:a")
    reg.register_model(only)
    reg.register_provider("stub", StubProvider())
    runtime = ModelRuntime(
        registry=reg,
        policy=DefaultModelPolicy(PolicyConstraints(generator_model_id=only.id)),
    )
    try:
        _run(runtime.invoke(ModelRequest(model="stub:a", prompt="hi")))
    except ModelPolicyRejected:
        pass
    else:
        raise AssertionError("expected ModelPolicyRejected")

    class Boom(StubProvider):
        async def invoke(self, request):
            raise RuntimeError("boom")

    reg2 = ModelRegistry()
    reg2.register_model(Model.from_ref("boom:a", provider="boom"))
    reg2.register_provider("boom", Boom(name="boom"))
    rec2 = InMemoryUsageRecorder()
    rt2 = ModelRuntime(registry=reg2, usage_recorder=rec2)
    try:
        _run(rt2.invoke(ModelRequest(model="boom:a", prompt="hi")))
    except RuntimeError:
        pass
    assert len(rec2.records) == 1 and rec2.records[0].success is False
    assert rec2.records[0].error_type == "RuntimeError"


def test_writer_result_and_usage_adapters():
    class FakeResult:
        output_text = "text"
        token_usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        model_name = "qwen3.6:27b"
        provider = "ollama"
        latency_ms = 12
        finish_reason = "stop"
        error = None
        raw_response = {}
        metadata = {}

    resp = MR.from_writer_result(FakeResult(), request_id="r", model_id="ollama:qwen3.6:27b")
    assert resp.usage.total_tokens == 5 and resp.latency_ms == 12
    usage = MU.from_token_usage({"prompt_tokens": 1, "completion_tokens": 1})
    assert usage.total_tokens == 2


def test_litellm_provider_delegates_to_legacy_service():
    calls = {}

    class FakeLLMService:
        def complete(self, prompt, *, context, agent_cfg, agent_name, llm_cfg_override=None):
            calls["prompt"] = prompt
            calls["override"] = llm_cfg_override
            return {"text": "legacy-out", "model_name": "qwen3", "cached": False}

    provider = LiteLLMProvider(FakeLLMService())
    req = ModelRequest(model=Model.from_ref("ollama:qwen3"), prompt="hello")
    resp = _run(provider.invoke(req))
    assert resp.output_text == "legacy-out"
    assert calls["prompt"] == "hello"
    assert calls["override"]["name"] == "qwen3"
    assert resp.provider == "ollama"  # resolved model provider, not adapter name
