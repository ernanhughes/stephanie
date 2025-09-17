# calibrators/quantile_fallback.py
import numpy as np


class QuantileThresholdCalibrator:
    def __init__(self, positive=True, quantile=0.85, slope=10.0):
        self.positive = positive
        self.quantile = quantile
        self.slope = slope
        self.thr = 0.5

    def fit(self, scores):
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            self.thr = 0.5
            return
        q = self.quantile if self.positive else (1.0 - self.quantile)
        self.thr = float(np.quantile(scores, q))

    def predict_proba(self, scores):
        scores = np.asarray(scores, dtype=float)
        z = (scores - self.thr) * self.slope
        p = 1.0 / (1.0 + np.exp(-z))
        if not self.positive:
            p = 1.0 - p
        return np.stack([1 - p, p], axis=1)  # [P(neg), P(pos)]
