# stephanie/portfolio/evaluator.py
"""Portfolio evaluation bridge (§10, sequence 3.2). Every response -> Stage 2."""
from __future__ import annotations

from typing import Any, Callable, Sequence

from stephanie.evaluation.criterion import Criterion
from stephanie.evaluation.evaluation import Evaluation, EvaluationObservation, EvaluatorRef
from stephanie.evaluation.score import Score
from stephanie.evaluation.subject import SubjectRef
from stephanie.portfolio.executor import PortfolioExecution

# score_fn(execution) -> list of (dimension, value, rationale) tuples.
ScoreFn = Callable[[PortfolioExecution], list[tuple[str, float, str | None]]]


class PortfolioEvaluator:
    def __init__(self, evaluation_runtime) -> None:
        self.evaluation_runtime = evaluation_runtime

    async def evaluate(
        self,
        executions: Sequence[PortfolioExecution],
        criterion: Criterion,
        evaluator: EvaluatorRef,
        score_fn: ScoreFn,
    ) -> list[Evaluation]:
        evaluations: list[Evaluation] = []
        for execution in executions:
            if not execution.success:
                continue
            observation = EvaluationObservation(
                subject=SubjectRef(
                    subject_type="model.response",
                    subject_id=execution.request_id or execution.execution_id,
                    text=execution.output_text,
                ),
                criterion=criterion,
                evaluator=evaluator,
                model_id=execution.model_id,
                task_type=execution.metadata.get("task_type"),
                scores=[
                    Score(score_id="", evaluation_id="", dimension=dim, value=value,
                          rationale=rationale, scorer=evaluator.name)
                    for dim, value, rationale in score_fn(execution)
                ],
                metadata={
                    "portfolio_candidate_id": execution.candidate_id,
                    "portfolio_role": execution.role.value,
                    "plan_id": execution.metadata.get("plan_id"),
                },
            )
            evaluations.append(
                await self.evaluation_runtime.record_from_model_invocation(
                    observation,
                    model_id=execution.model_id,
                    request_id=execution.request_id,
                    trace_id=execution.trace_id,
                    task_type=observation.task_type,
                    provider=execution.metadata.get("provider"),
                )
            )
        return evaluations
