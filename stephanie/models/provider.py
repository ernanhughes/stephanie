# stephanie/models/provider.py
"""Canonical provider abstraction (§6 — adapted from Writer's base.py).

Actual execution initially delegates to Stephanie's established
infrastructure (``services/llm_service.py``) via ``LiteLLMProvider``.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from stephanie.models.exceptions import ProviderInvocationError
from stephanie.models.model import Model
from stephanie.models.request import ModelRequest
from stephanie.models.response import ModelResponse
from stephanie.models.usage import ModelUsage


@dataclass(frozen=True)
class ProviderHealth:
    healthy: bool
    provider: str = ""
    detail: str = ""


class ModelProvider(ABC):
    @abstractmethod
    def supports(self, model: Model) -> bool:
        ...

    @abstractmethod
    async def invoke(self, request: ModelRequest) -> ModelResponse:
        ...


class SyncModelProvider(ModelProvider):
    """Synchronous transports bridged into the async contract."""

    @abstractmethod
    def invoke_sync(self, request: ModelRequest) -> ModelResponse:
        ...

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        return self.invoke_sync(request)


class StubProvider(ModelProvider):
    """Deterministic provider for contract tests (no I/O)."""

    def __init__(self, name: str = "stub", echo_prefix: str = "stub:"):
        self.name = name
        self.echo_prefix = echo_prefix

    def supports(self, model: Model) -> bool:
        return model.provider == self.name

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        model = request.model if isinstance(request.model, Model) else Model.from_ref(str(request.model))
        text = request.prompt or ""
        if request.messages:
            text = next(
                (m.content for m in reversed(request.messages) if m.role == "user"), text
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = ModelUsage(input_tokens=len(text.split()), output_tokens=1, latency_ms=latency_ms)
        return ModelResponse(
            request_id=str(uuid.uuid4()),
            model_id=model.id,
            provider=self.name,
            output_text=f"{self.echo_prefix}{text}",
            usage=usage,
            latency_ms=latency_ms,
            finish_reason="stop",
            created_at=datetime.utcnow(),
            metadata={"trace_id": request.trace_id} if request.trace_id else {},
        )


class LiteLLMProvider(ModelProvider):
    """Adapter over Stephanie's existing ``LLMService`` (Phase 2).

    Direction for the first pass (per spec §15)::

        ModelRuntime -> LiteLLMProvider -> existing llm_service

    so execution semantics do not change while the contract is established.
    """

    def __init__(self, llm_service: Any, provider_name: str = "litellm"):
        self._llm_service = llm_service
        self._provider_name = provider_name

    def supports(self, model: Model) -> bool:
        return True  # LiteLLM fans out; registry decides routing.

    async def health(self) -> ProviderHealth:
        try:
            result = self._llm_service.health_check()
            status = result.get("status", "")
            return ProviderHealth(
                healthy=status == "healthy",
                provider=self._provider_name,
                detail=status,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ProviderHealth(healthy=False, provider=self._provider_name, detail=str(exc))

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        import asyncio

        model = request.model if isinstance(request.model, Model) else Model.from_ref(str(request.model))
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        metadata = dict(model.metadata or {})
        llm_cfg_override = {
            "name": model.name,
            "api_base": metadata.get("api_base"),
            "api_key": metadata.get("api_key"),
        }
        context: dict[str, Any] = dict(request.metadata.get("context", {}) or {})
        agent_cfg: dict[str, Any] = dict(request.metadata.get("agent_cfg", {}) or {})
        agent_name = str(request.metadata.get("agent_name", request.task_type or "model_runtime"))

        def _call() -> dict[str, Any]:
            messages = request.to_messages()
            if len(messages) <= 1 and request.prompt is not None and not request.messages:
                return self._llm_service.complete(
                    request.prompt,
                    context=context,
                    agent_cfg=agent_cfg,
                    agent_name=agent_name,
                    llm_cfg_override=llm_cfg_override,
                )
            return self._llm_service.chat(
                messages,
                context=context,
                agent_cfg=agent_cfg,
                agent_name=agent_name,
                llm_cfg_override=llm_cfg_override,
            )

        try:
            result = await asyncio.to_thread(_call)
        except Exception as exc:
            raise ProviderInvocationError(self._provider_name, model.id, str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = ModelUsage(latency_ms=latency_ms)
        response = ModelResponse(
            request_id=request_id,
            model_id=model.id,
            provider=model.provider or self._provider_name,
            output_text=str(result.get("text", "")),
            usage=usage,
            latency_ms=latency_ms,
            finish_reason="stop" if not result.get("cached") else "cached",
            created_at=datetime.utcnow(),
            metadata={"trace_id": request.trace_id} if request.trace_id else {},
        )
        if result.get("cached"):
            response.metadata["cached"] = True
        return response
