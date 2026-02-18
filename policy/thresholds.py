# policy/thresholds.py

from dataclasses import dataclass


@dataclass
class ThresholdConfig:
    energy_safe: float = 0.35
    energy_warning: float = 0.45
    energy_critical: float = 0.55
    dominance_required: float = 0.80
