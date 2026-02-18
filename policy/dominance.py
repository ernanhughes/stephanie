from typing import Iterable

from policy.governance_signal import GovernanceSignal


class DominanceEngine:
    """
    True Pareto dominance check.
    """

    def __init__(self, critical_axes: Iterable[str]):
        self.critical_axes = set(critical_axes)

    def dominates(self, before: GovernanceSignal, after: GovernanceSignal) -> bool:
        improved_any = False

        for axis in self.critical_axes:
            before_val = getattr(before, axis)
            after_val = getattr(after, axis)

            if after_val < before_val:
                return False  # degraded critical axis

            if after_val > before_val:
                improved_any = True

        return improved_any
