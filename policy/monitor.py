from collections import deque
import numpy as np


class EnergySpiralDetector:

    def __init__(self, window: int = 100):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, energy: float):
        self.history.append(energy)

    def slope(self) -> float:
        if len(self.history) < self.window:
            return 0.0

        y = np.array(self.history)
        x = np.arange(len(y))
        return float(np.polyfit(x, y, 1)[0])

    def is_spiraling(self) -> bool:
        return self.slope() > 0.002
