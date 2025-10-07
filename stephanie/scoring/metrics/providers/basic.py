# stephanie/metrics/providers/basic.py
from .base import MetricProvider, MetricVector
from typing import Dict, Any, Tuple
import math

class BasicProvider:
    name = "basic"
    version = "1.0.0"

    async def compute(self, *, goal: str, text: str, context: Dict[str, Any]) -> MetricVector:
        g = (goal or "").strip().lower()
        t = (text or "").strip().lower()
        gs, ts = set(g.split()), set(t.split())
        # token overlap
        inter = len(gs & ts) if gs and ts else 0
        union = len(gs | ts) if gs or ts else 1
        jacc = inter/union
        # char overlap
        gc, tc = set(g), set(t)
        ci = len(gc & tc) if gc and tc else 0
        cu = len(gc | tc) if gc or tc else 1
        char = ci/cu

        names = (
            "basic.len_chars", "basic.len_tokens",
            "basic.goal_len_chars", "basic.goal_len_tokens",
            "basic.length_ratio", "basic.jaccard_token", "basic.char_overlap"
        )
        vals = (
            float(len(t)), float(len(ts)),
            float(len(g)), float(len(gs)),
            float(len(ts))/float(len(gs)) if len(gs) else 0.0,
            float(jacc), float(char)
        )
        return MetricVector(names, vals)
