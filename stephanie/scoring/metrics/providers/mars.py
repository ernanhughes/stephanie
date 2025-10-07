# stephanie/metrics/providers/mars.py
from .base import MetricProvider, MetricVector 
from .scorer import ScorerProvider
from typing import Dict, Any, Tuple
import math

def _sigmoid(x: float) -> float:
    try: return 1.0/(1.0+math.exp(-x))
    except OverflowError: return 0.0 if x < 0 else 1.0

class MARSProvider:
    name = "mars"
    version = "1.0.0"

    def __init__(self, mars_calculator=None, scorer_provider: ScorerProvider | None = None):
        self.mars = mars_calculator
        self.scorer_provider = scorer_provider  # reuse scorer outputs if needed

    async def compute(self, *, goal: str, text: str, context: Dict[str, Any]) -> MetricVector:
        try:
            # prefer dedicated calculator
            if self.mars and hasattr(self.mars, "calculate"):
                # assume scorer outputs already present in context or recompute quickly
                bundle = context.get("_scorer_bundle") or {}
                res = self.mars.calculate(bundle, context=context)
                # flatten to mars.*
                names = []; vals = []
                from .scorer import _flatten
                _flatten("mars", res or {}, names, vals)
                return MetricVector(tuple(names), tuple(vals))

            # fallback: fuse scorer vector in context if available
            flat_vals = context.get("_scorer_values")
            if not flat_vals: return MetricVector((), ())
            nums = [x for x in flat_vals if isinstance(x, (int, float))]
            if not nums: return MetricVector((), ())
            mean = sum(nums)/len(nums)
            var  = sum((x-mean)**2 for x in nums)/len(nums)
            smean = sum(_sigmoid(x) for x in nums)/len(nums)
            return MetricVector(("mars.consensus_mean","mars.consensus_var","mars.sigmoid_mean"),
                                (mean, var, smean))
        except Exception:
            return MetricVector((), ())
