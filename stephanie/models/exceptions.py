# stephanie/models/exceptions.py
"""Canonical exceptions for the Stephanie model runtime (Stage 1)."""
from __future__ import annotations


class ModelRuntimeError(Exception):
    """Base error for the model runtime."""


class ModelNotFound(ModelRuntimeError):
    """Raised when a model reference cannot be resolved."""


class ProviderNotFound(ModelRuntimeError):
    """Raised when no provider is registered for a model."""


class ProviderInvocationError(ModelRuntimeError):
    """Raised when a provider fails to produce a response."""

    def __init__(self, provider: str, model_id: str, message: str):
        super().__init__(f"[{provider}/{model_id}] {message}")
        self.provider = provider
        self.model_id = model_id


class ModelPolicyRejected(ModelRuntimeError):
    """Raised when policy rejects all routing candidates."""

    def __init__(self, reasons: list[str] | None = None):
        super().__init__("; ".join(reasons or ["rejected by policy"]))
        self.reasons = list(reasons or [])
