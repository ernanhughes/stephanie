# stephanie/scoring/base_evaluator.py
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, prompt: str, response: str = None) -> dict:
        """Returns a structured score dict with score, rationale, etc."""
        pass
