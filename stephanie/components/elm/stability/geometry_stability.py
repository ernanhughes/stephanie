# geometry_governor.py

from stephanie.data.score_bundle import ScoreBundle


class GeometryStabilityGovernor:

    def should_accept_update(
        self,
        before: ScoreBundle,
        after: ScoreBundle,
    ) -> bool:

        energy_before = before.get("hallucination_energy")
        energy_after = after.get("hallucination_energy")

        if energy_before and energy_after:
            if energy_after.score > 55.0:  # critical threshold
                return False

        # embedding variance check
        var = after.get("embedding_variance")
        if var and var.score < 30.0:
            return False

        return True
