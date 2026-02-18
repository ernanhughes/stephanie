from collections import deque
import numpy as np


class StabilityMonitor:

    def __init__(self, window=50, energy_spike=0.6):
        self.window = window
        self.energy_spike = energy_spike
        self.energy_history = deque(maxlen=window)
        self.dominance_history = deque(maxlen=window)

    def update(self, energy, dominance_ok):
        self.energy_history.append(energy)
        self.dominance_history.append(1 if dominance_ok else 0)

    def spiral_detected(self) -> bool:
        if len(self.energy_history) < self.window:
            return False

        mean_energy = np.mean(self.energy_history)
        dominance_rate = np.mean(self.dominance_history)

        if mean_energy > self.energy_spike:
            return True

        if dominance_rate < 0.6:
            return True

        return False
