# stephanie/models/usage.py
"""Normalized runtime measurements (§9 of the Stage 1 spec)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, MutableMapping, Optional


@dataclass
class ModelUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None

    total_tokens: Optional[int] = None

    estimated_cost_usd: Optional[Decimal] = None

    latency_ms: Optional[float] = None

    @staticmethod
    def from_token_usage(token_usage: Any, latency_ms: Optional[float] = None) -> "ModelUsage":
        """Normalize Writer-style ``ModelResult.token_usage`` dicts."""
        usage = ModelUsage(latency_ms=latency_ms)
        if isinstance(token_usage, dict):
            usage.input_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens"))
            usage.output_tokens = token_usage.get(
                "completion_tokens", token_usage.get("output_tokens")
            )
            usage.total_tokens = token_usage.get("total_tokens")
            if usage.total_tokens is None and (
                usage.input_tokens is not None or usage.output_tokens is not None
            ):
                usage.total_tokens = (usage.input_tokens or 0) + (usage.output_tokens or 0)
        return usage


@dataclass
class ModelInvocationRecord:
    request_id: str
    model_id: str
    provider: str

    task_type: Optional[str]

    usage: ModelUsage = field(default_factory=ModelUsage)

    success: bool = True
    error_type: Optional[str] = None

    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    metadata: MutableMapping[str, Any] = field(default_factory=dict)
