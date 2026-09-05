# stephanie/models/response.py
"""Canonical response envelope (§8 — adopts Writer's ModelResult contract)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, MutableMapping, Optional

from stephanie.models.usage import ModelUsage


@dataclass
class ModelResponse:
    request_id: str
    model_id: str
    provider: str

    output_text: str

    usage: ModelUsage = field(default_factory=ModelUsage)
    latency_ms: Optional[float] = None

    finish_reason: Optional[str] = None

    raw_response: Any | None = None

    created_at: Optional[datetime] = None

    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_writer_result(
        result: Any, *, request_id: str = "", model_id: str = "", trace_id: Optional[str] = None
    ) -> "ModelResponse":
        """Adapt Writer's ``ModelResult`` without forcing Writer to migrate."""
        usage = ModelUsage.from_token_usage(
            getattr(result, "token_usage", None),
            latency_ms=getattr(result, "latency_ms", None),
        )
        metadata = dict(getattr(result, "metadata", None) or {})
        if getattr(result, "error", None):
            metadata.setdefault("error", result.error)
        if trace_id:
            metadata.setdefault("trace_id", trace_id)
        return ModelResponse(
            request_id=request_id,
            model_id=model_id or getattr(result, "model_name", ""),
            provider=getattr(result, "provider", ""),
            output_text=getattr(result, "output_text", ""),
            usage=usage,
            latency_ms=getattr(result, "latency_ms", None),
            finish_reason=getattr(result, "finish_reason", None),
            raw_response=getattr(result, "raw_response", None),
            created_at=datetime.utcnow(),
            metadata=metadata,
        )
