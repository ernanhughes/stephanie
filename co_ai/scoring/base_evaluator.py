# co_ai/scoring/base_evaluator.py
from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, hypothesis: dict, context: dict = None) -> dict:
        """Returns a structured score dict with score, rationale, etc."""
        pass
