from typing import Any, Dict, List, Tuple

from policy.protocols.axes import AxisCalculator


class AxisEngine:

    def __init__(self, axes: List[AxisCalculator]):
        self.axes = axes

    def compute(
        self,
        claim_vec,
        evidence_vecs,
    ) -> Tuple[Dict[str, float], Any]:

        context: Dict[str, Any] = {
            "claim_vec": claim_vec,
            "evidence_vecs": evidence_vecs,
        }

        axes_values: Dict[str, float] = {}

        for axis in self.axes:
            value = axis.compute(context)
            axes_values[axis.name] = float(value)

        energy_result = context.get("energy_result")

        return axes_values, energy_result
