# co_ai/evaluator/base.py
from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, prompt, goal, output_a, output_b):
        pass
