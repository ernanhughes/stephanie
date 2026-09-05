# stephanie/portfolio/synthesis.py
"""Evidence-aware selection/synthesis (§15, §22). No majority voting, ever.

Hierarchy: deterministic evidence > external evidence > independent
supported reasoning > model agreement > raw vote count (forbidden).
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence
from uuid import uuid4

from stephanie.evaluation.evaluation import Evaluation
from stephanie.portfolio.disagreement import Disagreement
from stephanie.portfolio.executor import PortfolioExecution
from stephanie.portfolio.outcome import PortfolioOutcome
from stephanie.portfolio.verification import VerificationResult

SynthesizeFn = Callable[[str], str]  # structured brief -> synthesized text


class PortfolioSynthesizer:
    def __init__(self, synthesize_fn: Optional[SynthesizeFn] = None):
        self.synthesize_fn = synthesize_fn

    def select(
        self,
        executions: Sequence[PortfolioExecution],
        evaluations: Sequence[Evaluation],
        disagreements: Sequence[Disagreement],
        verifications: Sequence[VerificationResult],
        scores_by_evaluation: Mapping[str, Sequence] | None = None,
    ) -> PortfolioOutcome:
        """Select (not synthesize) the best candidate. Deterministic evidence first."""
        by_candidate = {e.candidate_id: e for e in executions if e.success}
        if not by_candidate:
            return self._outcome(None, executions, disagreements, verifications, None, "no successful executions")

        verified_pass = {
            self._execution_of_verification(v, executions)
            for v in verifications
            if v.passed is True
        } - {None}
        pool = verified_pass or set(by_candidate)
        if verified_pass:
            reason = "deterministic verification passed"
        else:
            reason = "highest evaluated quality among independently supported candidates"

        best_id, best_conf = self._rank(pool, evaluations, scores_by_evaluation)
        return self._outcome(best_id, executions, disagreements, verifications, best_conf, reason)

    def synthesize(
        self,
        executions: Sequence[PortfolioExecution],
        evaluations: Sequence[Evaluation],
        disagreements: Sequence[Disagreement],
        verifications: Sequence[VerificationResult],
        scores_by_evaluation: Mapping[str, Sequence] | None = None,
    ) -> PortfolioOutcome:
        """Evidence-aware synthesis: structured brief, never raw 'which is best?'."""
        selection = self.select(executions, evaluations, disagreements, verifications, scores_by_evaluation)
        if self.synthesize_fn is None:
            return selection
        brief = self._brief(executions, evaluations, disagreements, verifications, scores_by_evaluation)
        try:
            text = self.synthesize_fn(brief)
        except Exception as exc:
            return selection.__class__(**{**selection.__dict__, "metadata": {**selection.metadata, "synthesis_error": str(exc)}})
        return selection.__class__(
            **{**selection.__dict__, "synthesized_text": text,
               "metadata": {**selection.metadata, "synthesis_brief_chars": len(brief)}}
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _execution_of_verification(v: VerificationResult, executions):
        cid = (v.metadata or {}).get("candidate_id")
        for e in executions:
            if e.candidate_id == cid or v.claim_id and v.claim_id in (e.candidate_id, e.execution_id):
                return e.candidate_id
        return None

    @staticmethod
    def _rank(pool, evaluations, scores_by_evaluation):
        mean_by_candidate: dict[str, float] = {}
        eval_by_candidate = {e.metadata.get("portfolio_candidate_id"): e for e in evaluations}
        for cid in pool:
            evaluation = eval_by_candidate.get(cid)
            scores = list((scores_by_evaluation or {}).get(evaluation.evaluation_id, [])) if evaluation else []
            if scores:
                mean_by_candidate[cid] = sum(s.value for s in scores) / len(scores)
            else:
                mean_by_candidate[cid] = float("-inf")
        best_id = max(mean_by_candidate, key=lambda cid: (mean_by_candidate[cid], cid))
        best = mean_by_candidate[best_id]
        confidence = max(0.0, min(1.0, best)) if best != float("-inf") else None
        return best_id, confidence

    @staticmethod
    def _brief(executions, evaluations, disagreements, verifications, scores_by_evaluation) -> str:
        lines = ["PORTFOLIO SYNTHESIS BRIEF — decide from evidence, not votes.", ""]
        eval_by_candidate = {e.metadata.get("portfolio_candidate_id"): e for e in evaluations}
        for execution in executions:
            lines.append(f"## candidate {execution.candidate_id} ({execution.model_id}, {execution.role.value})")
            lines.append(execution.output_text[:2000])
            evaluation = eval_by_candidate.get(execution.candidate_id)
            if evaluation is not None:
                for score in (scores_by_evaluation or {}).get(evaluation.evaluation_id, []):
                    lines.append(f"- eval {score.dimension}: {score.value}")
            lines.append("")
        for disagreement in disagreements:
            lines.append(f"- disagreement [{disagreement.disagreement_type}] {disagreement.description}")
        for verification in verifications:
            lines.append(f"- verification [{verification.method}]: passed={verification.passed}")
        return "\n".join(lines)

    @staticmethod
    def _outcome(selected, executions, disagreements, verifications, confidence, reason):
        from decimal import Decimal

        latencies = [e.latency_ms for e in executions if e.latency_ms is not None]
        return PortfolioOutcome(
            outcome_id=f"out_{uuid4().hex[:10]}",
            plan_id=(executions[0].metadata.get("plan_id") if executions else None) or "",
            selected_candidate_id=selected,
            synthesized_text=None,
            confidence=confidence,
            candidate_ids=tuple(e.candidate_id for e in executions),
            disagreement_ids=tuple(d.disagreement_id for d in disagreements),
            verification_ids=tuple(v.verification_id for v in verifications),
            total_cost_usd=None,
            total_latency_ms=sum(latencies) if latencies else None,
            metadata={"selection_reason": reason},
        )
