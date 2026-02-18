# policy/protocols/scoring.py
from typing import Protocol
from stephanie.data.score_bundle import ScoreBundle

class ScoreProvider(Protocol):
    def evaluate(self, output: object, context: dict) -> ScoreBundle:
        ...
