from typing import Any
from .base import SignalProvider, SignalResult
from ..axes import RewardAxis


class CertumProvider(SignalProvider):
    def __init__(self, energy_model: Any):
        self.model = energy_model

    def compute(self, context_pack: Any, plan_trace: Any, output: Any, **kwargs) -> SignalResult:
        energy = float(self.model.compute_energy(output))
        failures = ["energy_spike"] if energy > 0.5 else []

        return SignalResult(
            axis_values={RewardAxis.HALLUCINATION_ENERGY: energy},
            diagnostics={"energy_raw": energy},
            failure_signatures=failures,
            confidence=1.0,
        )
