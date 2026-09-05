# stephanie/evaluation/provenance.py
"""Evaluation provenance (§15). prompt_hash stays provenance, never cache key."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class EvaluationProvenance:
    evaluator_name: str

    model_id: Optional[str] = None

    prompt_hash: Optional[str] = None
    config_hash: Optional[str] = None

    run_id: Optional[str] = None
    experiment_id: Optional[str] = None

    trace_id: Optional[str] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_model_invocation(
        *,
        evaluator_name: str,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        task_type: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> "EvaluationProvenance":
        """Link a Stage 1 ModelRuntime invocation to an evaluation (§32)."""
        metadata: dict[str, Any] = {}
        if request_id:
            metadata["model_request_id"] = request_id
        if task_type:
            metadata["task_type"] = task_type
        if provider:
            metadata["provider"] = provider
        return EvaluationProvenance(
            evaluator_name=evaluator_name,
            model_id=model_id,
            run_id=None,
            trace_id=trace_id,
            metadata=metadata,
        )
