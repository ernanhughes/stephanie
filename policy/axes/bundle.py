# certum/axes/bundle.py

from typing import Dict

from policy.utils.dict_utils import deep_get


class AxisBundle:
    def __init__(self, axes: Dict[str, float]):
        self._axes = axes

    def get(self, name: str) -> float:
        return self._axes.get(name, 0.0)

    def items(self):
        return self._axes.items()

    def __repr__(self):
        return f"AxisBundle({self._axes})"

    # -----------------------------------------------------
    # Factory: build from decision trace
    # -----------------------------------------------------

    @classmethod
    def from_trace(cls, row: Dict) -> "AxisBundle":
        """
        Reconstruct AxisBundle from full report row.
        """

        axes = {
            "energy": deep_get(row, "energy", "value"),
            "participation_ratio": deep_get(
                row, "energy", "geometry", "spectral", "participation_ratio"
            ),
            "sensitivity": deep_get(
                row, "energy", "geometry", "robustness", "sensitivity"
            ),
            "alignment": deep_get(
                row, "energy", "geometry", "alignment", "alignment_to_sigma1"
            ),
            "effectiveness": deep_get(row, "effectiveness"),
        }

        return cls(axes)
