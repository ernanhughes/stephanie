from .governance_signal import GovernanceSignal
from .dominance import DominanceEngine
from .calibration import EnergyRegimeEstimator
from .monitor import EnergySpiralDetector
from .decision import Decision


class BasePolicy:

    def __init__(
        self,
        dominance_engine: DominanceEngine,
        regime_estimator: EnergyRegimeEstimator,
        spiral_detector: EnergySpiralDetector,
    ):
        self.dominance_engine = dominance_engine
        self.regime_estimator = regime_estimator
        self.spiral_detector = spiral_detector

    def evaluate(
        self,
        current_signal: GovernanceSignal,
        previous_signal: GovernanceSignal | None = None,
    ) -> Decision:

        self.regime_estimator.update(current_signal.energy)
        self.spiral_detector.update(current_signal.energy)

        regime = self.regime_estimator.regime(current_signal.energy)

        if regime == "critical":
            return Decision.REJECT

        if self.spiral_detector.is_spiraling():
            return Decision.FREEZE

        if previous_signal:
            if not self.dominance_engine.dominates(previous_signal, current_signal):
                return Decision.REJECT

        return Decision.ACCEPT


