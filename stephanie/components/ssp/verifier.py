from __future__ import annotations
from typing import Tuple

class Verifier:
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.threshold = cfg.get("threshold", 0.6)

    @staticmethod
    def _f1_ref(text_a: str, text_b: str) -> float:
        a = text_a.lower().split()
        b = text_b.lower().split()
        if not a or not b:
            return 0.0
        inter = len(set(a) & set(b))
        p = inter / max(len(a), 1)
        r = inter / max(len(b), 1)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def verify(self, ground_truth: str, predicted: str) -> Tuple[bool, float]:
        score = self._f1_ref(ground_truth, predicted)
        return (score >= self.threshold), score
