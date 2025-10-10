# stephanie/arena/scoring/aggregate.py
from __future__ import annotations

from typing import Dict


class WeightedAggregator:
    def __init__(self, weights: Dict[str, float]):
        # weights per scorer name, multiply its "primary"
        self.w = dict(weights or {})

    def aggregate(self, per_scorer: Dict[str, Dict[str, float]]) -> float:
        # per_scorer: {scorer_name: {"primary": x, ...}}
        total = 0.0 
        denom = 0.0
        for name, sd in per_scorer.items():
            w = float(self.w.get(name, 1.0))
            total += w * float(sd.get("primary", 0.0))
            denom += abs(w)
        return total / denom if denom > 0 else 0.0
