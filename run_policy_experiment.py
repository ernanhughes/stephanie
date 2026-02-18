
from policy.calibration.adaptive_calibrator import CalibrationResult
from policy.experiments.runaway_report import RunawayReportHarness
from policy.policy_container import PolicyContainer
import numpy as np
from typing import Callable, Dict, List, Optional, Any

# -----------------------
# Dummy AI
# -----------------------

def stephanie_sim(state):
    return {
        "claim": f"quality={state['quality']}",
        "evidence": ["some evidence text"]
    }


# -----------------------
# Dummy Energy Function
# -----------------------

def dummy_energy(output, context):
    q = float(output["claim"].split("=")[1])
    return abs(q - 1.0)


# -----------------------
# Dummy Policy (accept low energy)
# -----------------------

class SimplePolicy:
    def __init__(self, tau=0.2):
        self.tau = tau

    def decide(self, energy):
        return energy < self.tau

import numpy as np


class QuantileCalibrator:
    """
    Adaptive quantile-based calibrator with drift detection.

    - Learns tau from energy distribution
    - Tracks mean energy
    - Provides drift signal
    """

    def __init__(self, quantile=0.2, warmup=50, drift_window=50):
        self.quantile = quantile
        self.warmup = warmup
        self.drift_window = drift_window

        self.history = []
        self.tau_accept = None

        self.energy_mean = None
        self.energy_std = None

    # ----------------------------------
    # Update calibration
    # ----------------------------------

    def update(self, energy: float):
        energy = float(energy)
        self.history.append(energy)

        if len(self.history) >= self.warmup:
            arr = np.array(self.history)

            self.tau_accept = float(
                np.quantile(arr, self.quantile)
            )

            self.energy_mean = float(np.mean(arr))
            self.energy_std = float(np.std(arr))

    # ----------------------------------
    # Margin score (used by policy)
    # ----------------------------------

    def margin_score(self, energy: float, calibration=None) -> float:
        if self.tau_accept is None:
            return 0.0
        return float(self.tau_accept - float(energy))

    # ----------------------------------
    # Drift detection
    # ----------------------------------

    def detect_drift(self, recent_history, calibration=None, drift_threshold=3.0) -> bool:
        """
        Compare recent window mean to calibration mean.
        """

        if self.energy_mean is None:
            return 0.0

        recent = np.array(recent_history[-self.drift_window:])
        if len(recent) == 0:
            return 0.0

        mean_recent = float(np.mean(recent))

        drift = abs(mean_recent - self.energy_mean)

        if self.energy_std and self.energy_std > 1e-6:
            return drift / self.energy_std

        return drift


calibrator = QuantileCalibrator(
    quantile=0.2,   # allow 20% lowest-energy samples
    warmup=50       # wait before activating
)

policy = PolicyContainer(
    ai_callable=stephanie_sim,
    energy_function=dummy_energy,
    calibrator=calibrator,
    calibration=calibrator,  # use same object
)

harness = RunawayReportHarness(
    ai_callable=stephanie_sim,
    energy_function=dummy_energy,
    policy_container=policy,
    episodes=1000,
)

report = harness.run()
print(report)
