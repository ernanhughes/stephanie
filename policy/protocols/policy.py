# certum/protocols/policy.py

from typing import Optional, Protocol

from policy.axes.bundle import AxisBundle
from policy.custom_types import Verdict


class Policy(Protocol):

    # Required attributes
    tau_accept: float
    tau_review: Optional[float]
    hard_negative_gap: float

    # Required behavior
    def decide(
        self,
        axes: AxisBundle,
        effectiveness_score: float
    ) -> Verdict:
        ...

    @property
    def name(self) -> str:
        ...
