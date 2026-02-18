axis_configfrom enum import Enum
from dataclasses import dataclass


class AxisDirection(str, Enum):
    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


@dataclass
class AxisSpec:
    direction: AxisDirection
    weight: float = 1.0
