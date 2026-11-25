# components/elm/axes.py

from enum import Enum
from typing import Dict


class AxisDirection(str, Enum):
    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


# Canonical axis semantics keyed by axis name.
AXIS_SEMANTICS: Dict[str, AxisDirection] = {
    "hrm_alignment": AxisDirection.HIGHER_IS_BETTER,
    "hallucination_energy": AxisDirection.LOWER_IS_BETTER,
    "embedding_margin": AxisDirection.HIGHER_IS_BETTER,
    "policy_advantage": AxisDirection.HIGHER_IS_BETTER,
    "metric_alignment": AxisDirection.HIGHER_IS_BETTER,
    "coherence": AxisDirection.HIGHER_IS_BETTER,
    "context_fidelity": AxisDirection.HIGHER_IS_BETTER,
    # Dimension-native aliases used by the governed ELM harness.
    # The experiment harness works directly on ScoreBundle dimensions
    # (alignment / energy / margin), not on the older RewardAxis enum.
    "alignment": AxisDirection.HIGHER_IS_BETTER,
    "reasoning_quality": AxisDirection.HIGHER_IS_BETTER,
    "energy": AxisDirection.LOWER_IS_BETTER,
    "speculation": AxisDirection.LOWER_IS_BETTER,
    "margin": AxisDirection.HIGHER_IS_BETTER,
    "similarity": AxisDirection.HIGHER_IS_BETTER,
    "advantage": AxisDirection.HIGHER_IS_BETTER,
    "context_grounding": AxisDirection.HIGHER_IS_BETTER,
}

# Backward-compat mapping: ScoreBundle dimension -> governance axis name.
# Older code imported this from signal_extractor; it now lives here.
DIMENSION_TO_AXIS: Dict[str, str] = {
    "alignment": "alignment",
    "reasoning_quality": "alignment",
    "energy": "energy",
    "speculation": "energy",
    "margin": "margin",
    "similarity": "margin",
    "advantage": "advantage",
    "context_grounding": "context_fidelity",
    # Pass through canonical axis names.
    "hrm_alignment": "hrm_alignment",
    "hallucination_energy": "hallucination_energy",
    "embedding_margin": "embedding_margin",
    "policy_advantage": "policy_advantage",
    "metric_alignment": "metric_alignment",
    "coherence": "coherence",
    "context_fidelity": "context_fidelity",
}

__all__ = ["AxisDirection", "AXIS_SEMANTICS", "DIMENSION_TO_AXIS"]
