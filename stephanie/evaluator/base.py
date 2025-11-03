# stephanie/evaluator/base.py
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def judge(self, prompt, output_a, output_b, context: dict) -> dict:
        pass

    @abstractmethod
    def score_single(self, prompt, output, context: dict) -> float:
        pass
