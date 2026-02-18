# policy/protocols/governance.py

from typing import Protocol
from stephanie.data.score_bundle import ScoreBundle

class GovernanceEngine(Protocol):
    def assess(self, bundle: ScoreBundle) -> dict:
        """
        Returns governance metrics:
            {
                "energy": float,
                "regime": str,
                "dominates": bool,
                ...
            }
        """
        ...
