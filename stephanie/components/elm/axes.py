# components/elm/axes.py

from enum import Enum


class AxisDirection(str, Enum):
    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


AXIS_SEMANTICS = {
    "hrm_alignment": AxisDirection.HIGHER_IS_BETTER,
    "hallucination_energy": AxisDirection.LOWER_IS_BETTER,
    "embedding_margin": AxisDirection.HIGHER_IS_BETTER,
    "policy_advantage": AxisDirection.HIGHER_IS_BETTER,
    "metric_alignment": AxisDirection.HIGHER_IS_BETTER,
    "coherence": AxisDirection.HIGHER_IS_BETTER,
    "context_fidelity": AxisDirection.HIGHER_IS_BETTER,
}
