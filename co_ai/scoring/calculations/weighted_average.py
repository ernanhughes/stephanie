# co_ai/scoring/calculations/weighted_average.py
from co_ai.scoring.calculations.base_calculator import BaseScoreCalculator
from co_ai.scoring.score_bundle import ScoreBundle


class WeightedAverageCalculator(BaseScoreCalculator):
    def calculate(self, bundle: ScoreBundle) -> float:
        results = bundle.results.values()
        total = sum(r.score * getattr(r, "weight", 1.0) for r in results)
        weight_sum = sum(getattr(r, "weight", 1.0) for r in results)
        return round(total / weight_sum, 2) if weight_sum else 0.0
