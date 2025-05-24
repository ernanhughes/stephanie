# co_ai/evaluator/base.py
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def judge(self, prompt, goal, output_a, output_b):
        pass
