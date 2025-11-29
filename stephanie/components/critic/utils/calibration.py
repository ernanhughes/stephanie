# stephanie/components/critic/utils/calibration.py
from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

BinStrategy = Literal["uniform", "quantile"]

def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
    strategy: BinStrategy = "uniform",
) -> Dict[str, np.ndarray]:
    """
    Compute per-bin confidence (mean prob), accuracy (mean label), count.
    strategy='uniform' → equal-width bins in [0,1]
    strategy='quantile' → equal-size bins by empirical quantiles
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    p = np.clip(p, 0, 1)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(p, edges, right=True) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    elif strategy == "quantile":
        # Ensure distinct edges; handle duplicates
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(p, qs))
        # guarantee at least 2 edges
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
        bin_ids = np.digitize(p, edges, right=True) - 1
        bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
        n_bins = len(edges) - 1
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    conf = np.zeros(n_bins, dtype=float)
    acc  = np.zeros(n_bins, dtype=float)
    cnt  = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        idx = (bin_ids == b)
        cnt[b] = int(idx.sum())
        if cnt[b] > 0:
            conf[b] = float(p[idx].mean())
            acc[b]  = float(y[idx].mean())
        else:
            conf[b] = np.nan
            acc[b]  = np.nan

    return {
        "bin_edges": edges,
        "bin_confidence": conf,
        "bin_accuracy": acc,
        "bin_count": cnt,
    }

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
    strategy: BinStrategy = "uniform",
) -> float:
    """
    ECE = sum_b (n_b / N) * |acc_b - conf_b|
    Skips empty bins.
    """
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    conf = bins["bin_confidence"]
    acc  = bins["bin_accuracy"]
    cnt  = bins["bin_count"].astype(float)
    N = cnt.sum()
    if N <= 0:
        return float("nan")
    mask = cnt > 0
    ece = np.sum((cnt[mask] / N) * np.abs(acc[mask] - conf[mask]))
    return float(ece)

def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Wrapper for Brier score with NaN safety."""
    try:
        return float(brier_score_loss(y_true, np.clip(y_prob, 0, 1)))
    except Exception:
        return float("nan")

# ----------------------------
# Optional: simple calibrators
# ----------------------------

class PlattCalibrator:
    """Logistic calibration on probabilities."""
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        y = np.asarray(y_true).astype(int)
        p = np.clip(np.asarray(y_prob, dtype=float), 0, 1).reshape(-1, 1)
        self.clf.fit(p, y)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(y_prob, dtype=float), 0, 1).reshape(-1, 1)
        return self.clf.predict_proba(p)[:, 1]

class IsoCalibrator:
    """Isotonic regression calibration."""
    def __init__(self, y_min: float = 0.0, y_max: float = 1.0):
        self.iso = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip")

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        y = np.asarray(y_true).astype(int)
        p = np.clip(np.asarray(y_prob, dtype=float), 0, 1)
        self.iso.fit(p, y)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(y_prob, dtype=float), 0, 1)
        return self.iso.transform(p)
