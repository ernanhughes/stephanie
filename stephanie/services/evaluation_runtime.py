# stephanie/services/evaluation_runtime.py
"""Orchestrating Evaluation Runtime (§22).

Coordinates subject -> evaluator -> scores -> evidence -> confidence ->
interpretation -> persistence. Never silently fuses unless asked.
Links Stage 1 ModelRuntime invocations via provenance (§32).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from stephanie.evaluation.compare import ComparisonOutcome, compare_evaluations
from stephanie.evaluation.diagnostics import DUAL_WRITE_FAILURE, EvaluationDiagnostic
from stephanie.evaluation.evaluation import Evaluation, EvaluationObservation
from stephanie.evaluation.fusion import FusedScore, FusionSpec, fuse_weighted_mean
from stephanie.evaluation.provenance import EvaluationProvenance
from stephanie.evaluation.repository import InMemoryEvaluationRepository
from stephanie.evaluation.score import Score
from stephanie.evaluation.subject import SubjectRef
from stephanie.evaluation.writer import append_observation


class EvaluationRuntime:
    def __init__(self, repository: Optional[InMemoryEvaluationRepository] = None):
        self.repository = repository or InMemoryEvaluationRepository()
        self.diagnostics: list[EvaluationDiagnostic] = []

    async def record(self, observation: EvaluationObservation) -> Evaluation:
        """Persist an observation with its raw scores, attributes, evidence."""
        evaluation = await append_observation(self.repository, observation)
        if observation.attributes:
            await self.repository.add_evaluation_attributes(observation.attributes)
        if observation.score_attributes:
            await self.repository.add_score_attributes(observation.score_attributes)
        if observation.evidence:
            from stephanie.evaluation.evidence import EvaluationEvidenceLink

            for evidence in observation.evidence:
                await self.repository.link_evidence(
                    EvaluationEvidenceLink(
                        evaluation_id=evaluation.evaluation_id,
                        evidence_id=evidence.evidence_id,
                        relationship=observation.metadata.get("evidence_relationship", "supports"),
                    )
                )
        if observation.provenance is not None:
            evaluation.metadata["provenance"] = _provenance_to_dict(observation.provenance)
        return evaluation

    async def record_from_model_invocation(
        self,
        observation: EvaluationObservation,
        *,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        task_type: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Evaluation:
        """Attach Stage 1 model-invocation provenance (§32) then record."""
        from stephanie.evaluation.context import EvaluationContext

        context = EvaluationContext.from_model_response(
            task_type=task_type,
            request_id=request_id,
            trace_id=trace_id,
            model_id=model_id,
            provider=provider,
        )
        context.apply_to(observation, observation.evaluator.name)
        return await self.record(observation)

    async def fuse(
        self, evaluation_id: str, spec: FusionSpec
    ) -> FusedScore:
        """Explicit, recomputable fusion over raw scores (§16)."""
        scores = await self.repository.scores(evaluation_id)
        return fuse_weighted_mean(scores, spec)

    async def dual_write(
        self,
        observation: EvaluationObservation,
        legacy_write_fn,
    ) -> tuple[Evaluation, bool, bool]:
        """Shadow-write: legacy + canonical tracked separately (§29).

        Canonical failure never fails production: returns
        (canonical_evaluation_or_raised, legacy_success, canonical_success).
        """
        legacy_success = True
        try:
            await legacy_write_fn(observation)
        except Exception as exc:
            legacy_success = False
            self.diagnostics.append(
                EvaluationDiagnostic(DUAL_WRITE_FAILURE, f"legacy write failed: {exc}")
            )
        canonical_success = True
        evaluation = None
        try:
            evaluation = await self.record(observation)
        except Exception as exc:
            canonical_success = False
            self.diagnostics.append(
                EvaluationDiagnostic(DUAL_WRITE_FAILURE, f"canonical write failed: {exc}")
            )
            raise
        return evaluation, legacy_success, canonical_success


def _provenance_to_dict(provenance: Any) -> dict:
    if hasattr(provenance, "__dict__") or hasattr(provenance, "__dataclass_fields__"):
        from dataclasses import asdict, is_dataclass

        if is_dataclass(provenance):
            return asdict(provenance)
    return dict(provenance or {})
