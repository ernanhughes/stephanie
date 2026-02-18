import numpy as np
from collections import deque


class EnergyRegimeEstimator:

    def __init__(self, window: int = 500):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, energy: float):
        self.history.append(energy)

    def percentile(self, p: float) -> float:
        if not self.history:
            return 1.0
        return float(np.percentile(self.history, p))

    def regime(self, energy: float) -> str:
        if len(self.history) < 50:
            return "uncalibrated"

        p90 = self.percentile(90)
        p98 = self.percentile(98)

        if energy > p98:
            return "critical"
        elif energy > p90:
            return "warning"
        return "stable"


class QuantileEnergyCalibrator:
    """
    FAR-controlled adaptive thresholding.

    tau = Q_negatives(FAR_target)
    """

    def __init__(self, far_target: float = 0.01):
        self.far_target = far_target
        self.tau = None

    def fit(self, negative_energies):
        self.tau = float(np.quantile(negative_energies, self.far_target))

    def accept(self, energy: float) -> bool:
        if self.tau is None:
            raise RuntimeError("Calibrator not fitted")
        return energy <= self.tau


