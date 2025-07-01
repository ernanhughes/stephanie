# stephanie/scoring/calculations/mrq_normalizer.py
"""
MRQNormalizerCalculator

This class is used to normalize raw MR.Q scores into a standardized range (typically 0–100)
to support consistent evaluation across dimensions. It takes expected min and max scores as
bounds, then rescales incoming scores accordingly. This is useful when MR.Q scoring output
is unstable or on a different scale than LLM-based scoring.

Key Functions:
- normalize raw scores per dimension
- apply clipping to keep scores within [0, 1] before scaling
- compute a weighted average as the final score
- used by ScoringManager to align MR.Q scores with other evaluators
All right let's stay away for a bit mine
Intended for use in the scoring pipeline to support adaptive tuning and fair comparisons.
"""


from stephanie.scoring.calculations.base_calculator import BaseCalculator


class MRQNormalizerCalculator(BaseCalculator):
    def __init__(self, expected_min=0.0, expected_max=1.0, clip=True, scale=100.0):
        self.expected_min = expected_min
        self.expected_max = expected_max
        self.clip = clip
        self.scale = scale  # typically 1.0 or 100.0

    def calculate(self, results: dict) -> float:
        raw_total = 0.0
        weight_sum = 0.0

        for dim, val in results.items():
            raw = val["score"]
            norm = (raw - self.expected_min) / max((self.expected_max - self.expected_min), 1e-6)

            if self.clip:
                norm = max(0.0, min(norm, 1.0))

            val["normalized_score"] = round(norm * self.scale, 2)

            raw_total += norm * self.scale * val.get("weight", 1.0)
            weight_sum += val.get("weight", 1.0)

        return round(raw_total / weight_sum, 2) if weight_sum else 0.0
