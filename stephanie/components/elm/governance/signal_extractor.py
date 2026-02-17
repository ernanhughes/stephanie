from typing import Dict, List, Optional
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult


class GovernanceSignalExtractor:
    """
    Governance logic operating directly on ScoreBundle.

    No RewardVector.
    No static axis enums.
    Fully dimension-native.
    """

    def __init__(
        self,
        *,
        critical_dimensions: Optional[List[str]] = None,
        lower_is_better: Optional[List[str]] = None,
    ):
        # Dimensions that must strictly improve
        self.critical_dimensions = critical_dimensions or [
            "alignment",
            "energy",
            "margin",
        ]

        # Direction semantics per dimension
        self.lower_is_better = set(lower_is_better or ["energy"])

    # -------------------------------------------------------
    # Core Signal Extraction
    # -------------------------------------------------------

    def extract_metrics(self, bundle: ScoreBundle) -> Dict[str, float]:
        """
        Extract normalized governance metrics from ScoreBundle.

        Returns flat dict of:
            dimension -> normalized_score (0-1)
            plus raw energy if available
        """
        metrics: Dict[str, float] = {}

        for dim, result in bundle.results.items():
            norm = self._normalize_score(result.score)

            if dim in self.lower_is_better:
                norm = 1.0 - norm  # invert so higher always = better

            metrics[dim] = norm

            # Extract raw energy if available
            if dim == "energy":
                raw_energy = self._extract_raw_energy(result)
                if raw_energy is not None:
                    metrics["energy_raw"] = raw_energy

        return metrics

    # -------------------------------------------------------
    # Dominance
    # -------------------------------------------------------

    def dominates(
        self,
        before: ScoreBundle,
        after: ScoreBundle,
    ) -> bool:
        """
        Strict dominance on critical dimensions.
        """

        diff = after.diff(before)

        for dim in self.critical_dimensions:
            dim_data = diff.get("dimensions", {}).get(dim)
            if not dim_data:
                return False

            delta = dim_data.get("score_delta", 0.0)

            if dim in self.lower_is_better:
                # improvement means score decreased
                if delta >= 0:
                    return False
            else:
                if delta <= 0:
                    return False

        return True

    # -------------------------------------------------------
    # Direction-Normalized Delta Vector
    # -------------------------------------------------------

    def delta_vector(
        self,
        before: ScoreBundle,
        after: ScoreBundle,
    ) -> Dict[str, float]:
        """
        Direction-aware delta vector.
        Positive always means improvement.
        """

        diff = after.diff(before)
        out: Dict[str, float] = {}

        for dim, dim_data in diff.get("dimensions", {}).items():
            delta = dim_data.get("score_delta")
            if delta is None:
                continue

            # Normalize assuming 0–100 scoring
            norm_delta = float(delta) / 100.0

            if dim in self.lower_is_better:
                norm_delta = -norm_delta

            out[dim] = norm_delta

        return out

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------

    @staticmethod
    def _normalize_score(score: float) -> float:
        """
        Normalize assuming 0–100 scoring range.
        """
        try:
            x = float(score) / 100.0
            return max(0.0, min(1.0, x))
        except Exception:
            return 0.0

    @staticmethod
    def _extract_raw_energy(result: ScoreResult) -> Optional[float]:
        """
        Extract raw_energy from attributes if present.
        """
        attrs = getattr(result, "attributes", None)
        if isinstance(attrs, dict):
            raw = attrs.get("raw_energy")
            if raw is not None:
                try:
                    return float(raw)
                except Exception:
                    return None
        return None
