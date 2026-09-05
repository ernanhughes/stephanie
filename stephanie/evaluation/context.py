# stephanie/evaluation/context.py
"""Shared causal-chain contract (§Stage 2.5).

One chain, always reconstructable::

    task -> request_id -> model_id/provider -> response
      -> evaluation_id -> score_ids -> evidence_ids -> fusion

``trace_id`` (Stage 1 ModelRuntime) and ``run_id`` are the cross-system
link keys. Nothing here invents new IDs — it threads existing ones through.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from stephanie.evaluation.evaluation import EvaluationObservation
from stephanie.evaluation.provenance import EvaluationProvenance


@dataclass(frozen=True)
class EvaluationContext:
    """Carries the invocation chain into an evaluation observation."""

    task_type: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    run_id: Optional[str] = None
    model_id: Optional[str] = None
    provider: Optional[str] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_model_response(
        *,
        task_type: Optional[str] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> "EvaluationContext":
        return EvaluationContext(
            task_type=task_type,
            request_id=request_id,
            trace_id=trace_id,
            run_id=run_id,
            model_id=model_id,
            provider=provider,
        )

    def apply_to(self, observation: EvaluationObservation, evaluator_name: str) -> EvaluationObservation:
        """Fill identity/provenance gaps on an observation without overwriting caller values."""
        if observation.task_type is None:
            observation.task_type = self.task_type
        if observation.model_id is None:
            observation.model_id = self.model_id
        if observation.run_id is None:
            observation.run_id = self.run_id
        if observation.provenance is None:
            observation.provenance = EvaluationProvenance.from_model_invocation(
                evaluator_name=evaluator_name,
                model_id=observation.model_id,
                request_id=self.request_id,
                trace_id=self.trace_id or self.request_id,
                task_type=observation.task_type,
                provider=self.provider,
            )
        return observation

    def chain_summary(self, *, evaluation_id: Optional[str] = None) -> dict:
        return {
            "task_type": self.task_type,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "evaluation_id": evaluation_id,
        }
