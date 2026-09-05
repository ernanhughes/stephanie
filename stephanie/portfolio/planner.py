# stephanie/portfolio/planner.py
"""Deterministic portfolio planner (§7). History alters defaults only in 3.8."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import uuid4

from stephanie.models.model import Model
from stephanie.portfolio.candidate import PortfolioCandidate
from stephanie.portfolio.diagnostics import (
    BUDGET_EXCEEDED,
    NO_ELIGIBLE_MODEL,
    ROLE_UNFILLED,
    PortfolioDiagnostic,
)
from stephanie.portfolio.plan import PortfolioBudget, PortfolioPlan
from stephanie.portfolio.policy import PortfolioPolicy
from stephanie.portfolio.roles import PortfolioRole


class PortfolioPlanner:
    def __init__(self, available_models: list[Model] | None = None):
        self.available_models = list(available_models or [])

    async def plan(
        self,
        request,  # ModelRequest — the primary task
        policy: PortfolioPolicy,
    ) -> PortfolioPlan:
        from stephanie.models.request import ModelRequest as _MR

        candidates: list[PortfolioCandidate] = []
        diagnostics: list[str] = []

        eligible = [m for m in self.available_models if m.enabled]
        if policy.preferred_primary:
            preferred = [m for m in eligible if m.id == policy.preferred_primary]
            if preferred:
                eligible = preferred + [m for m in eligible if m.id != policy.preferred_primary]
        if not eligible:
            raise PortfolioDiagnostic(NO_ELIGIBLE_MODEL, "no enabled models available")

        primary_model = eligible[0]
        candidates.append(
            self._candidate(
                request, primary_model, PortfolioRole.PRIMARY, "primary_generation",
                reason=f"strongest eligible model ({primary_model.id})",
            )
        )

        if policy.include_independent_reviewer:
            reviewer = self._different_provider(eligible, primary_model) if policy.require_different_provider_for_reviewer else self._next(eligible, primary_model)
            if reviewer is None:
                diagnostics.append(f"{ROLE_UNFILLED}: independent_reviewer")
            else:
                candidates.append(
                    self._candidate(
                        request, reviewer, PortfolioRole.INDEPENDENT_REVIEWER,
                        "independent_review_a",
                        reason=f"different provider ({reviewer.provider}) for independence",
                    )
                )

        if policy.include_breadth:
            breadth = self._cheapest(eligible, exclude={primary_model.id} | {c.model_id for c in candidates})
            if breadth is None:
                breadth = self._next(eligible, primary_model)
            if breadth is None:
                diagnostics.append(f"{ROLE_UNFILLED}: breadth")
            else:
                candidates.append(
                    self._candidate(
                        request, breadth, PortfolioRole.BREADTH, "independent_review_b",
                        reason=f"cheap/different breadth ({breadth.id})",
                    )
                )

        # Synthesizer reuses the primary model (no extra provider needed).
        candidates.append(
            self._candidate(
                request, primary_model, PortfolioRole.SYNTHESIZER, "anchored_synthesis",
                reason=f"synthesis via primary ({primary_model.id})",
            )
        )

        trimmed, budget_notes = self._apply_budget(candidates, policy.budget)
        diagnostics.extend(budget_notes)

        return PortfolioPlan(
            plan_id=f"plan_{uuid4().hex[:12]}",
            task_type=policy.task_type,
            candidates=tuple(trimmed),
            budget=policy.budget,
            synthesis_policy=policy.synthesis_policy,
            created_at=datetime.utcnow(),
            metadata={"diagnostics": diagnostics, "policy_task": policy.task_type},
        )

    def _candidate(self, request, model: Model, role, group: str, reason: str) -> PortfolioCandidate:
        return PortfolioCandidate(
            candidate_id=f"{role.value}_{uuid4().hex[:8]}",
            model_id=model.id,
            role=role,
            request=replace(request, model=model),
            independence_group=group,
            metadata={"reason": reason, "provider": model.provider},
        )

    @staticmethod
    def _different_provider(eligible: list[Model], primary: Model) -> Model | None:
        for model in eligible:
            if model.id != primary.id and model.provider != primary.provider:
                return model
        return None

    @staticmethod
    def _next(eligible: list[Model], primary: Model) -> Model | None:
        for model in eligible:
            if model.id != primary.id:
                return model
        return None

    @staticmethod
    def _cheapest(eligible: list[Model], exclude: set[str]) -> Model | None:
        # Stage 3.1 proxy for cheap: local models first, then first eligible.
        for model in eligible:
            if model.id not in exclude and model.local:
                return model
        for model in eligible:
            if model.id not in exclude:
                return model
        return None

    @staticmethod
    def _apply_budget(
        candidates: list[PortfolioCandidate], budget: PortfolioBudget
    ) -> tuple[list[PortfolioCandidate], list[str]]:
        notes: list[str] = []
        if budget.max_models is not None and len(candidates) > budget.max_models:
            # Priority: PRIMARY, INDEPENDENT_REVIEWER, BREADTH, SYNTHESIZER.
            order = {
                PortfolioRole.PRIMARY: 0,
                PortfolioRole.INDEPENDENT_REVIEWER: 1,
                PortfolioRole.BREADTH: 2,
                PortfolioRole.SYNTHESIZER: 3,
            }
            ranked = sorted(candidates, key=lambda c: order.get(c.role, 9))
            dropped = ranked[budget.max_models:]
            notes.append(
                f"{BUDGET_EXCEEDED}: dropped {[c.role.value for c in dropped]} (max_models={budget.max_models})"
            )
            return ranked[: budget.max_models], notes
        return candidates, notes
