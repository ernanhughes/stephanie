# stephanie/services/model_runtime.py
"""Orchestrating service for the canonical model runtime (§14).

Internal pipeline::

    request -> resolve -> registry -> router -> policy ->
    provider -> normalize -> record usage -> return
"""
from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime
from typing import Any, Optional

from stephanie.models.exceptions import ModelPolicyRejected
from stephanie.models.model import Model
from stephanie.models.policy import DefaultModelPolicy, ModelPolicy
from stephanie.models.pricing import NullPricingService, PricingService
from stephanie.models.provider import ModelProvider
from stephanie.models.registry import ModelRegistry
from stephanie.models.request import ModelRequest
from stephanie.models.response import ModelResponse
from stephanie.models.routing import ModelRouter, RoutingContext, SimpleModelRouter
from stephanie.models.usage import ModelInvocationRecord


class InMemoryUsageRecorder:
    """Phase 5 observation store (no routing optimization yet)."""

    def __init__(self) -> None:
        self.records: list[ModelInvocationRecord] = []

    async def record(
        self, *, request: ModelRequest, response: ModelResponse, success: bool = True,
        error_type: Optional[str] = None,
    ) -> ModelInvocationRecord:
        model = request.model if isinstance(request.model, Model) else None
        record = ModelInvocationRecord(
            request_id=response.request_id,
            model_id=response.model_id,
            provider=response.provider,
            task_type=request.task_type,
            usage=response.usage,
            success=success,
            error_type=error_type,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            metadata={"trace_id": request.trace_id} if request.trace_id else {},
        )
        self.records.append(record)
        return record


class ModelRuntime:
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        router: Optional[ModelRouter] = None,
        policy: Optional[ModelPolicy] = None,
        pricing: Optional[PricingService] = None,
        usage_recorder: Optional[InMemoryUsageRecorder] = None,
    ):
        self.registry = registry or ModelRegistry()
        self.router = router or SimpleModelRouter(self.registry)
        self.policy = policy or DefaultModelPolicy()
        self.pricing = pricing or NullPricingService()
        self.usage_recorder = usage_recorder or InMemoryUsageRecorder()

    def register_provider(self, name: str, provider: ModelProvider) -> None:
        self.registry.register_provider(name, provider)

    def register_model(self, model: Model) -> None:
        self.registry.register_model(model)

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        if not request.trace_id:
            request = replace(request, trace_id=str(uuid.uuid4()))

        context = RoutingContext.from_request(request)
        candidates = self.router.route(request, context)
        decision = self.policy.evaluate(request, candidates)
        if not decision.allowed:
            raise ModelPolicyRejected(decision.reasons)

        model = self.select_model(candidates, decision)
        provider = self.registry.provider_for(model)
        namespaced = request.with_model(model)

        try:
            response = await provider.invoke(namespaced)
        except Exception as exc:
            await self.usage_recorder.record(
                request=namespaced,
                response=ModelResponse(
                    request_id=request.trace_id or str(uuid.uuid4()),
                    model_id=model.id,
                    provider=model.provider,
                    output_text="",
                    created_at=datetime.utcnow(),
                ),
                success=False,
                error_type=type(exc).__name__,
            )
            raise

        # Fill gaps providers leave blank, then price (unknown stays unknown).
        if not response.request_id:
            response.request_id = request.trace_id or str(uuid.uuid4())
        if response.latency_ms is None:
            response.latency_ms = response.usage.latency_ms
        if response.usage.latency_ms is None:
            response.usage.latency_ms = response.latency_ms
        try:
            response.usage.estimated_cost_usd = self.pricing.estimate(model, response.usage)
        except Exception:
            pass
        if request.trace_id:
            response.metadata.setdefault("trace_id", request.trace_id)

        await self.usage_recorder.record(request=namespaced, response=response)
        return response

    def select_model(self, candidates: list[Model], decision: Any) -> Model:
        preferred = getattr(decision, "preferred_models", []) or []
        by_id = {m.id: m for m in candidates}
        for model_id in preferred:
            if model_id in by_id:
                return by_id[model_id]
        excluded = set(getattr(decision, "excluded_models", []) or [])
        for candidate in candidates:
            if candidate.id not in excluded:
                return candidate
        return candidates[0]

    # -- Compatibility: legacy Stephanie callers -------------------------

    def complete_compat(
        self,
        prompt: str,
        *,
        model: str = "ollama:qwen3",
        task_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ModelRequest:
        """Build a ModelRequest from a legacy ``complete(prompt)`` call."""
        return ModelRequest(model=model, prompt=prompt, task_type=task_type, metadata=metadata or {})
