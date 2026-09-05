# stephanie/services/portfolio_runtime.py
"""Portfolio orchestrator (§1). Task -> plan -> independent execution ->
evaluation -> disagreement -> verification -> synthesis -> learning observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from stephanie.evaluation.criterion import Criterion
from stephanie.evaluation.evaluation import Evaluation
from stephanie.models.request import ModelRequest
from stephanie.portfolio.disagreement import Disagreement, DisagreementAnalyzer
from stephanie.portfolio.evaluator import PortfolioEvaluator
from stephanie.portfolio.executor import PortfolioExecution, PortfolioExecutor
from stephanie.portfolio.outcome import PortfolioOutcome
from stephanie.portfolio.plan import PortfolioPlan
from stephanie.portfolio.planner import PortfolioPlanner
from stephanie.portfolio.policy import PortfolioPolicy
from stephanie.portfolio.synthesis import PortfolioSynthesizer
from stephanie.portfolio.verification import VerificationResult, VerifierRegistry


@dataclass
class PortfolioResult:
    plan: PortfolioPlan
    executions: list[PortfolioExecution]
    evaluations: list[Evaluation]
    disagreements: list[Disagreement]
    verifications: list[VerificationResult]
    outcome: PortfolioOutcome
    scores_by_evaluation: dict[str, list] = field(default_factory=dict)


class PortfolioRuntime:
    def __init__(
        self,
        planner: PortfolioPlanner,
        executor: PortfolioExecutor,
        evaluator: PortfolioEvaluator,
        synthesizer: PortfolioSynthesizer | None = None,
        disagreement_analyzer: DisagreementAnalyzer | None = None,
        verifier_registry: VerifierRegistry | None = None,
    ):
        self.planner = planner
        self.executor = executor
        self.evaluator = evaluator
        self.synthesizer = synthesizer or PortfolioSynthesizer()
        self.disagreement_analyzer = disagreement_analyzer or DisagreementAnalyzer()
        self.verifier_registry = verifier_registry or VerifierRegistry()

    async def run(
        self,
        request: ModelRequest,
        policy: PortfolioPolicy,
        criterion,
        evaluator_ref,
        score_fn,
        *,
        synthesize: bool = False,
    ) -> PortfolioResult:
        plan = await self.planner.plan(request, policy)
        executions = await self.executor.execute(plan)
        evaluations = await self.evaluator.evaluate(executions, criterion, evaluator_ref, score_fn)
        scores_by_evaluation: dict[str, list] = {}
        reader = getattr(self.evaluator.evaluation_runtime, "repository", None)
        if reader is not None:
            for evaluation in evaluations:
                scores_by_evaluation[evaluation.evaluation_id] = list(
                    await reader.scores(evaluation.evaluation_id)
                )
        disagreements = self.disagreement_analyzer.analyze(executions, evaluations, scores_by_evaluation)
        verifications = self.verifier_registry.verify(executions, disagreements)
        if synthesize:
            outcome = self.synthesizer.synthesize(
                executions, evaluations, disagreements, verifications, scores_by_evaluation
            )
        else:
            outcome = self.synthesizer.select(
                executions, evaluations, disagreements, verifications, scores_by_evaluation
            )
        return PortfolioResult(
            plan=plan,
            executions=executions,
            evaluations=evaluations,
            disagreements=disagreements,
            verifications=verifications,
            outcome=outcome,
            scores_by_evaluation=scores_by_evaluation,
        )
