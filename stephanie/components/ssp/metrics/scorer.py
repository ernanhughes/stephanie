# stephanie/components/ssp/metrics/scorer.py
from __future__ import annotations
from typing import Dict, Any

from stephanie.components.ssp.metrics.calculator import SSPMetricsCalculator
from stephanie.components.ssp.metrics.scorable import SSPScorable

class SSPScorer:
    """Uniform interface: returns names, values, vector, version."""
    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.calculator = SSPMetricsCalculator(cfg)

    def score(self, scorable: SSPScorable) -> Dict[str, Any]:
        mv = self.calculator.score(scorable)
        return {
            "version": mv.version,
            "names": mv.names,
            "values": mv.values,
            "vector": mv.vector,
        }
