# stephanie/models/routing.py
"""Select candidate model resources (§12).

Returns ``list[Model]`` (usually one element in Stage 1) so
``route(task) -> portfolio`` can emerge without changing the contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Optional

from stephanie.models.model import Model
from stephanie.models.registry import ModelRegistry
from stephanie.models.request import ModelRequest


@dataclass
class RoutingContext:
    task_type: Optional[str] = None
    required_capabilities: set[str] = field(default_factory=set)

    preferred_model: Optional[str] = None

    latency_budget_ms: Optional[float] = None
    cost_budget_usd: Optional[Decimal] = None

    metadata: Mapping[str, object] = field(default_factory=dict)

    @staticmethod
    def from_request(request: ModelRequest) -> "RoutingContext":
        preferred = request.model.id if isinstance(request.model, Model) else request.model
        required: set[str] = set()
        if request.tools:
            required.add("tool_use")
        if request.response_schema is not None:
            required.add("structured_output")
        return RoutingContext(
            task_type=request.task_type,
            required_capabilities=required,
            preferred_model=preferred,
        )


class ModelRouter(ABC):
    @abstractmethod
    def route(self, request: ModelRequest, context: RoutingContext) -> list[Model]:
        ...


class SimpleModelRouter(ModelRouter):
    """Stage 1 router: preferred model first, else first capable model."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def route(self, request: ModelRequest, context: RoutingContext) -> list[Model]:
        if context.preferred_model:
            return [self.registry.resolve(context.preferred_model)]
        candidates = self.registry.available_models()
        if context.required_capabilities:
            capable = [
                m for m in candidates if m.capabilities.satisfies(context.required_capabilities)
            ]
            if capable:
                candidates = capable
        if not candidates:
            # No registered models: resolve ad-hoc so execution can proceed.
            fallback = request.model if isinstance(request.model, str) else request.model.id
            return [self.registry.resolve(fallback or "ollama:qwen3")]
        return [candidates[0]]
