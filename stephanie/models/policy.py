# stephanie/models/policy.py
"""Decide whether a technically available route is acceptable (§13)."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Sequence

from stephanie.models.model import Model
from stephanie.models.request import ModelRequest


@dataclass
class ModelPolicyDecision:
    allowed: bool

    preferred_models: list[str] = field(default_factory=list)
    excluded_models: list[str] = field(default_factory=list)

    escalate: bool = False

    reasons: list[str] = field(default_factory=list)


class ModelPolicy:
    def evaluate(
        self, request: ModelRequest, candidates: Sequence[Model]
    ) -> ModelPolicyDecision:
        raise NotImplementedError


@dataclass
class PolicyConstraints:
    local_only: bool = False
    max_cost_usd: Optional[Decimal] = None
    generator_model_id: Optional[str] = None  # critic-must-differ rule
    allow_escalation: bool = True


class DefaultModelPolicy(ModelPolicy):
    """Stage 1 policy: local-only, capability, and critic-differs rules."""

    def __init__(self, constraints: Optional[PolicyConstraints] = None):
        self.constraints = constraints or PolicyConstraints()

    def evaluate(
        self, request: ModelRequest, candidates: Sequence[Model]
    ) -> ModelPolicyDecision:
        constraints = self._constraints_for(request)
        allowed = [m for m in candidates if m.enabled]
        reasons: list[str] = []

        if constraints.local_only:
            before = len(allowed)
            allowed = [m for m in allowed if m.local]
            if len(allowed) < before:
                reasons.append("non-local models excluded (local-only)")

        if constraints.generator_model_id:
            before = len(allowed)
            allowed = [m for m in allowed if m.id != constraints.generator_model_id]
            if len(allowed) < before:
                reasons.append(
                    f"generator {constraints.generator_model_id} excluded (critic-must-differ)"
                )

        if not allowed:
            return ModelPolicyDecision(allowed=False, reasons=reasons or ["no candidates survive policy"])

        escalate = False
        if request.metadata.get("require_frontier_escalation") and constraints.allow_escalation:
            escalate = True
            reasons.append("frontier escalation requested")

        return ModelPolicyDecision(
            allowed=True,
            preferred_models=[m.id for m in allowed],
            escalate=escalate,
            reasons=reasons,
        )

    def _constraints_for(self, request: ModelRequest) -> PolicyConstraints:
        meta = request.metadata or {}
        base = self.constraints
        generator = meta.get("generator_model_id", base.generator_model_id)
        local_only = bool(meta.get("local_only", base.local_only))
        return PolicyConstraints(
            local_only=local_only,
            max_cost_usd=meta.get("max_cost_usd", base.max_cost_usd),
            generator_model_id=generator,
            allow_escalation=bool(meta.get("allow_escalation", base.allow_escalation)),
        )
