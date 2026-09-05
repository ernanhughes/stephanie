# stephanie/models/registry.py
"""One source of truth for available models and providers (§11)."""
from __future__ import annotations

from typing import Optional

from stephanie.models.capability import ModelCapabilities
from stephanie.models.exceptions import ModelNotFound, ProviderNotFound
from stephanie.models.model import Model, split_model_ref
from stephanie.models.provider import ModelProvider


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, Model] = {}
        self._providers: dict[str, ModelProvider] = {}

    def register_model(self, model: Model) -> None:
        self._models[model.id] = model

    def register_provider(self, name: str, provider: ModelProvider) -> None:
        self._providers[name] = provider

    def get_model(self, model_id: str) -> Model:
        try:
            return self._models[model_id]
        except KeyError:
            raise ModelNotFound(model_id) from None

    def get_provider(self, provider: str) -> ModelProvider:
        try:
            return self._providers[provider]
        except KeyError:
            raise ProviderNotFound(provider) from None

    def resolve(self, model_ref: str) -> Model:
        """Resolve ``"ollama:qwen3"``, ``"opencode-go:..."`` etc. to a Model.

        Unknown references become ad-hoc ``Model`` values (compatibility
        input format) rather than errors, so existing callers keep working.
        """
        ref = (model_ref or "").strip()
        if ref in self._models:
            return self._models[ref]
        provider, name = split_model_ref(ref)
        canonical_id = f"{provider}:{name}"
        if canonical_id in self._models:
            return self._models[canonical_id]
        return Model.from_ref(ref)

    def provider_for(self, model: Model) -> ModelProvider:
        if model.provider in self._providers:
            return self._providers[model.provider]
        # Fallback: first provider claiming support (e.g. LiteLLM fan-out).
        for provider in self._providers.values():
            try:
                if provider.supports(model):
                    return provider
            except Exception:
                continue
        raise ProviderNotFound(model.provider)

    def available_models(self, *, capability: Optional[str] = None) -> list[Model]:
        models = [m for m in self._models.values() if m.enabled]
        if capability:
            models = [m for m in models if getattr(m.capabilities, capability, False)]
        return models
