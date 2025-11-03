# stephanie/components/jitter/trends.py
from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np


class TrendTracker:
    """
    Maintains short/medium-term trend stats over recent VPM composite scores.
    Outputs: ewma, slope, volatility, novelty (z-score vs. recent history).
    """
    def __init__(self, window:int=200, ewma_alpha:float=0.1):
        self.window = window
        self.ewma_alpha = ewma_alpha
        self.buf = deque(maxlen=window)
        self._ewma = None

    def update(self, x: float) -> Dict[str, float]:
        x = float(np.clip(x, 0.0, 1.0))
        self.buf.append(x)
        # EWMA
        self._ewma = x if self._ewma is None else (self.ewma_alpha*x + (1-self.ewma_alpha)*self._ewma)
        # slope via simple regression on last N
        y = np.array(self.buf, dtype=np.float32)
        n = y.size
        slope = 0.0
        if n >= 5:
            t = np.arange(n, dtype=np.float32)
            t = (t - t.mean()) / (t.std() + 1e-8)
            slope = float(np.dot(y - y.mean(), t) / (np.sum(t*t) + 1e-8))
        vol = float(y.std()) if n >= 2 else 0.0
        z = float((x - y.mean()) / (y.std() + 1e-8)) if n >= 10 else 0.0
        return {"ewma": float(self._ewma), "slope": slope, "vol": vol, "novelty": z}
