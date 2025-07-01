# stephanie/scoring/calculations/base_calculator.py
from abc import ABC, abstractmethod

from stephanie.scoring.score_bundle import ScoreBundle


class BaseScoreCalculator(ABC):
    @abstractmethod
    def calculate(self, results: ScoreBundle) -> float:
        """
        Given a dict of dimension results (each with score, weight), return a single float score.
        """
        pass
