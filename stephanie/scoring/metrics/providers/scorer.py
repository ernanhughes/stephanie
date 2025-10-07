# stephanie/metrics/providers/scorer.py
from .base import MetricProvider, MetricVector
from typing import Dict, Any, Tuple

def _flatten(prefix: str, obj: Any, out_names, out_vals):
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            _flatten(f"{prefix}.{k}" if prefix else k, obj[k], out_names, out_vals)
    elif isinstance(obj, bool):
        out_names.append(prefix); out_vals.append(1.0 if obj else 0.0)
    elif isinstance(obj, (int, float)):
        out_names.append(prefix); out_vals.append(float(obj))
    # ignore strings/None/complex types

class ScorerProvider:
    name = "scorer"
    version = "1.0.0"

    def __init__(self, scoring):
        self.scoring = scoring  # your multi-scorer hub

    async def compute(self, *, goal: str, text: str, context: Dict[str, Any]) -> MetricVector:
        if not self.scoring:
            return MetricVector((), ())
        try:
            res = self.scoring.score_text(text, goal=goal, context=context)
            res = await res if hasattr(res, "__await__") else res
            names, vals = [], []
            _flatten("scorer", res or {}, names, vals)
            return MetricVector(tuple(names), tuple(vals))
        except Exception:
            return MetricVector((), ())
