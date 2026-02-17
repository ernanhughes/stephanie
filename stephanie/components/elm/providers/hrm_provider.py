from typing import Any
from .base import SignalProvider, SignalResult
from ..axes import RewardAxis


class HRMProvider(SignalProvider):
    def __init__(self, hrm_model: Any):
        self.model = hrm_model

    def compute(self, context_pack: Any, plan_trace: Any, output: Any, **kwargs) -> SignalResult:
        score = float(self.model.score(output))

        return SignalResult(
            axis_values={RewardAxis.HRM_ALIGNMENT: score},
            diagnostics={"hrm_raw": score},
            confidence=0.9,
        )
