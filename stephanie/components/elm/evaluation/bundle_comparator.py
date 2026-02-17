# components/elm/dominance/bundle_comparator.py

from typing import List, Dict
from stephanie.data.score_bundle import ScoreBundle
from components.elm.axes import AXIS_SEMANTICS, AxisDirection


class BundleComparator:

    @staticmethod
    def delta(
        before: ScoreBundle,
        after: ScoreBundle,
    ) -> Dict[str, float]:
        """
        Direction-normalized delta.
        Positive = improvement.
        """

        deltas: Dict[str, float] = {}

        dims = set(before.results.keys()) | set(after.results.keys())

        for dim in dims:
            b = before.get(dim)
            a = after.get(dim)

            if not b or not a:
                continue

            direction = AXIS_SEMANTICS.get(dim, AxisDirection.HIGHER_IS_BETTER)

            if direction == AxisDirection.HIGHER_IS_BETTER:
                delta = a.score - b.score
            else:
                delta = b.score - a.score

            deltas[dim] = delta

        return deltas

    @staticmethod
    def dominates(
        before: ScoreBundle,
        after: ScoreBundle,
        critical_axes: List[str],
    ) -> bool:
        """
        Strict Pareto dominance on critical axes.
        """

        for dim in critical_axes:
            b = before.get(dim)
            a = after.get(dim)

            if not b or not a:
                return False

            direction = AXIS_SEMANTICS.get(dim, AxisDirection.HIGHER_IS_BETTER)

            if direction == AxisDirection.HIGHER_IS_BETTER:
                if a.score <= b.score:
                    return False
            else:
                if a.score >= b.score:
                    return False

        return True
