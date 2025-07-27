# stephanie/agents/analysis/cost_benefit_analyzer.py

import logging
from statistics import mean

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class CostBenefitAnalyzerAgent(BaseAgent):
    """
    Evaluates the trade-off between model accuracy (alignment to LLM) and
    cost (time, compute, instability) for each inference layer.

    Produces an 'efficiency score' per layer for dynamic pipeline optimization.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger or logging.getLogger(__name__))

        self.min_evals = cfg.get("min_evaluations", 10)
        self.include_dimensions = cfg.get("include_dimensions", ["alignment"])
        self.cost_penalty_weight = cfg.get("cost_penalty_weight", 1.0)
        self.allowed_evaluators = ["mrq", "ebt", "svm", "gild", "llm"]

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, {})
        goal_id = goal.get("id")
        if not goal_id:
            self.logger.warning("Missing goal ID in context.")
            return context

        efficiency_table = self._compute_efficiency_scores(goal_id)
        context["efficiency_scores"] = efficiency_table
        return context

    def _compute_efficiency_scores(self, goal_id: int) -> dict:
        """
        For each evaluator (e.g., MRQ, EBT), compute average alignment to LLM,
        cost metrics, and a normalized efficiency score.
        """
        evaluations = self.memory.evaluations.get_by_goal(goal_id)
        by_layer = {}

        for e in evaluations:
            if e.evaluator_name not in self.allowed_evaluators:
                continue
            if not e.scores or not e.embedding_type:
                continue

            if e.evaluator_name not in by_layer:
                by_layer[e.evaluator_name] = {
                    "alignment_scores": [],
                    "times": [],
                    "uncertainty": [],
                    "count": 0,
                    "embedding_type": e.embedding_type,
                }

            score_bundle = by_layer[e.evaluator_name]
            dims = [s for s in e.dimension_scores if s.dimension in self.include_dimensions]
            for s in dims:
                if s.score is not None:
                    score_bundle["alignment_scores"].append(s.score)
                if s.uncertainty is not None:
                    score_bundle["uncertainty"].append(s.uncertainty)
                score_bundle["count"] += 1

                # Placeholder: capture cost from `extra_data`
                score_bundle["times"].append(
                    e.extra_data.get("duration", 0.0) if e.extra_data else 0.0
                )

        return self._normalize_scores(by_layer)

    def _normalize_scores(self, table: dict) -> dict:
        """
        Returns efficiency = alignment / (time + uncertainty)
        """
        result = {}
        for evaluator, metrics in table.items():
            if metrics["count"] < self.min_evals:
                continue

            alignment = mean(metrics["alignment_scores"]) if metrics["alignment_scores"] else 0.0
            time_cost = mean(metrics["times"]) if metrics["times"] else 1.0
            uncertainty = mean(metrics["uncertainty"]) if metrics["uncertainty"] else 0.0

            cost = time_cost + (uncertainty * self.cost_penalty_weight)
            efficiency = alignment / cost if cost > 0 else 0.0

            result[evaluator] = {
                "efficiency": round(efficiency, 4),
                "avg_alignment": round(alignment, 4),
                "avg_time": round(time_cost, 4),
                "avg_uncertainty": round(uncertainty, 4),
                "embedding_type": metrics["embedding_type"],
                "eval_count": metrics["count"],
            }

        return result
