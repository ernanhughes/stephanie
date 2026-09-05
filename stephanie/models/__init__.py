# stephanie/models/__init__.py
"""Canonical model runtime (Stage 1).

Pipeline: ModelRequest -> Routing -> Policy -> Provider ->
ModelResponse -> Usage recording -> Outcome/evaluation linkage.
"""
from __future__ import annotations

from stephanie.models.capability import ModelCapabilities
from stephanie.models.exceptions import (
    ModelNotFound,
    ModelPolicyRejected,
    ModelRuntimeError,
    ProviderInvocationError,
    ProviderNotFound,
)
from stephanie.models.model import LLAMA_CPP_PREFIXES, OPENCODE_PREFIXES, Model, ModelSpecAdapter
from stephanie.models.policy import DefaultModelPolicy, ModelPolicy, ModelPolicyDecision, PolicyConstraints
from stephanie.models.pricing import NullPricingService, PriceEntry, PricingService, StaticPricingService
from stephanie.models.provider import LiteLLMProvider, ModelProvider, ProviderHealth, StubProvider, SyncModelProvider
from stephanie.models.registry import ModelRegistry
from stephanie.models.request import ModelMessage, ModelRequest, ToolDefinition
from stephanie.models.response import ModelResponse
from stephanie.models.routing import ModelRouter, RoutingContext, SimpleModelRouter
from stephanie.models.usage import ModelInvocationRecord, ModelUsage

__all__ = [
    "Model",
    "ModelCapabilities",
    "ModelMessage",
    "ModelPolicy",
    "ModelPolicyDecision",
    "DefaultModelPolicy",
    "PolicyConstraints",
    "ModelProvider",
    "SyncModelProvider",
    "StubProvider",
    "LiteLLMProvider",
    "ProviderHealth",
    "ModelRegistry",
    "ModelRequest",
    "ToolDefinition",
    "ModelResponse",
    "ModelRouter",
    "RoutingContext",
    "SimpleModelRouter",
    "ModelUsage",
    "ModelInvocationRecord",
    "PricingService",
    "NullPricingService",
    "StaticPricingService",
    "PriceEntry",
    "ModelSpecAdapter",
    "ModelRuntimeError",
    "ModelNotFound",
    "ProviderNotFound",
    "ProviderInvocationError",
    "ModelPolicyRejected",
    "LLAMA_CPP_PREFIXES",
    "OPENCODE_PREFIXES",
]
