import math
import random

class VPMAdapter:
    """
    Minimal adapter: .diversity()->[0..1], .mutate_rate (rw), .active_count()->int
    Replace with Memcube/your VPM manager.
    """
    def __init__(self):
        self.mutate_rate = 0.05
        self._active = 400

    def diversity(self) -> float:
        # simulate diversity oscillation
        return 0.6 + 0.3 * math.sin(random.random() * 3.14)

    def active_count(self) -> int:
        return self._active
