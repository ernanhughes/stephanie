# stephanie/portfolio/executor.py
"""Portfolio executor (§8–§9). Independence is enforced structurally.

INDEPENDENT_ROLES receive the original task only. ANCHORED_ROLES (critic,
verifier, synthesizer) receive task + referenced candidate answers.
It is impossible to accidentally anchor an INDEPENDENT_REVIEWER: the
anchoring path requires an explicit ``anchor_candidate_ids`` argument
and rejects independent roles outright.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence
from uuid import uuid4

from stephanie.portfolio.candidate import PortfolioCandidate
from stephanie.portfolio.diagnostics import (
    EXECUTION_FAILED,
    INDEPENDENCE_VIOLATION,
    PortfolioDiagnostic,
)
from stephanie.portfolio.plan import PortfolioPlan
from stephanie.portfolio.roles import ANCHORED_ROLES, INDEPENDENT_ROLES, PortfolioRole


@dataclass(frozen=True)
class PortfolioExecution:
    execution_id: str
    candidate_id: str
    model_id: str
    role: PortfolioRole

    output_text: str

    request_id: str
    trace_id: str | None

    latency_ms: float | None
    usage: Any = None

    success: bool = True
    error: str | None = None

    anchored_on: tuple[str, ...] = ()
    created_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PortfolioExecutor:
    def __init__(self, model_runtime) -> None:
        self.model_runtime = model_runtime

    async def execute(self, plan: PortfolioPlan) -> list[PortfolioExecution]:
        by_group = self._group_independent(plan.candidates)
        results: list[PortfolioExecution] = []
        for group in by_group:
            # Candidates in different independence groups run concurrently;
            # anchored candidates run after their anchors resolve.
            group_results = await asyncio.gather(
                *(self._invoke(candidate, plan, {}) for candidate in group),
                return_exceptions=True,
            )
            for candidate, result in zip(group, group_results):
                if isinstance(result, Exception):
                    results.append(
                        PortfolioExecution(
                            execution_id=f"exec_{uuid4().hex[:10]}",
                            candidate_id=candidate.candidate_id,
                            model_id=candidate.model_id,
                            role=candidate.role,
                            output_text="",
                            request_id="",
                            trace_id=None,
                            latency_ms=None,
                            success=False,
                            error=f"{EXECUTION_FAILED}: {result}",
                            created_at=datetime.utcnow(),
                        )
                    )
                else:
                    results.append(result)
        return results

    async def execute_anchored(
        self,
        candidate: PortfolioCandidate,
        prior_executions: Sequence[PortfolioExecution],
        anchor_candidate_ids: Sequence[str],
    ) -> PortfolioExecution:
        """Anchored critique: allowed ONLY for ANCHORED_ROLES (§8)."""
        if candidate.role in INDEPENDENT_ROLES:
            raise PortfolioDiagnostic(
                INDEPENDENCE_VIOLATION,
                f"role {candidate.role.value} must never receive anchored answers",
                {"candidate_id": candidate.candidate_id},
            )
        if candidate.role not in ANCHORED_ROLES:
            raise PortfolioDiagnostic(
                INDEPENDENCE_VIOLATION,
                f"role {candidate.role.value} is not an anchored role",
                {"candidate_id": candidate.candidate_id},
            )
        anchors = {e.candidate_id: e for e in prior_executions}
        missing = [cid for cid in anchor_candidate_ids if cid not in anchors]
        if missing:
            raise PortfolioDiagnostic(
                EXECUTION_FAILED, f"anchor candidates not executed: {missing}"
            )
        context = "\n\n".join(
            f"--- candidate {cid} ({anchors[cid].model_id}) ---\n{anchors[cid].output_text}"
            for cid in anchor_candidate_ids
        )
        anchored_prompt = (
            f"{candidate.request.prompt or ''}\n\n"
            f"Candidate answers under review:\n{context}"
        )
        return await self._invoke(candidate, None, {}, override_prompt=anchored_prompt,
                                  anchored_on=tuple(anchor_candidate_ids))

    async def _invoke(self, candidate, plan, _unused, override_prompt=None, anchored_on=()):
        from dataclasses import replace

        request = candidate.request
        if override_prompt is not None:
            request = replace(request, prompt=override_prompt)
        response = await self.model_runtime.invoke(request)
        return PortfolioExecution(
            execution_id=f"exec_{uuid4().hex[:10]}",
            candidate_id=candidate.candidate_id,
            model_id=response.model_id,
            role=candidate.role,
            output_text=response.output_text,
            request_id=response.request_id,
            trace_id=request.trace_id,
            latency_ms=response.latency_ms,
            usage=response.usage,
            anchored_on=tuple(anchored_on),
            created_at=datetime.utcnow(),
            metadata={"plan_id": plan.plan_id if plan else None,
                      "independence_group": candidate.independence_group,
                      "task_type": request.task_type,
                      "provider": candidate.metadata.get("provider")},
        )

    @staticmethod
    def _group_independent(
        candidates: Sequence[PortfolioCandidate],
    ) -> list[list[PortfolioCandidate]]:
        # One wave per independence group; synthesizer runs last (own group).
        groups: dict[str, list[PortfolioCandidate]] = {}
        for candidate in candidates:
            groups.setdefault(candidate.independence_group or "default", []).append(candidate)
        ordered = sorted(groups.items(), key=lambda kv: (kv[0] != "primary_generation", kv[0]))
        return [members for _, members in ordered]
