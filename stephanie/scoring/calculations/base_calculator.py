# stephanie/scoring/calculations/base_calculator.py
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseScoreCalculator(ABC):
    @abstractmethod
    def calculate(self, results) -> float:
        """
        Given a dict of dimension results (each with score, weight), return a single float score.
        """
        pass
