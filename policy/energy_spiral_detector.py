
from collections import deque
import numpy as np

class EnergySpiralDetector:

    def __init__(self, window=100):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, energy):
        self.history.append(energy)

    def is_spiraling(self):
        if len(self.history) < self.window:
            return False

        y = np.array(self.history)
        x = np.arange(len(y))

        slope = np.polyfit(x, y, 1)[0]

        return slope > 0.002  # upward drift
