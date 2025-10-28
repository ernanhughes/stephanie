# stephanie/models/entropy_calibrator.py
from __future__ import annotations

from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class EntropyCalibrator:
    """
    Dual-mode calibrator:
      - Logistic regression (fast, interpretable)
      - Conformal prediction (guaranteed coverage)

    Features: [mean_entropy, high_entropy_tile_pct, entropy_variance]
    """

    def __init__(self, model_type="conformal", model_path="models/entropy_calibrator.pkl"):
        self.model_type = model_type
        self.model_path = model_path
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray, calibrate_size: float = 0.2):
        if self.model_type == "logistic":
            self.model = LogisticRegression()
            self.model.fit(X, y)

        elif self.model_type == "conformal":
            # Split calibration set
            n_cal = int(len(X) * calibrate_size)
            idx = np.random.permutation(len(X))
            X_train, y_train = X[idx[n_cal:]], y[idx[n_cal:]]
            X_cal, y_cal = X[idx[:n_cal]], y[idx[:n_cal]]

            # Train base model
            self.model = RandomForestClassifier()
            self.model.fit(X_train, y_train)

            # Nonconformity scores
            probs = self.model.predict_proba(X_cal)[:, 1]
            self.nonconformity_scores = np.abs(y_cal - probs)

        self.is_trained = True

    def get_threshold(self, target_coverage: float = 0.85) -> float:
        if self.model_type == "logistic":
            raise RuntimeError("Use predict_proba directly for logistic")
        elif self.model_type == "conformal":
            alpha = 1 - target_coverage
            n = len(self.nonconformity_scores)
            q = np.ceil((n + 1) * (1 - alpha)) / n
            return np.quantile(self.nonconformity_scores, q, method="higher")

    def predict_trust(self, X: np.ndarray, target_coverage: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.model.predict_proba(X)[:, 1]

        if self.model_type == "logistic":
            # Threshold by percentile of dev probs
            threshold = np.percentile(probs, 100 * (1 - target_coverage))
            return probs > threshold, probs

        elif self.model_type == "conformal":
            threshold = self.get_threshold(target_coverage)
            nonconformity = np.abs(1 - probs)
            return nonconformity <= threshold, probs

    def save(self):
        joblib.dump({
            "model_type": self.model_type,
            "model": self.model,
            "nonconformity_scores": getattr(self, "nonconformity_scores", None)
        }, self.model_path)

    def load(self):
        data = joblib.load(self.model_path)
        self.model_type = data["model_type"]
        self.model = data["model"]
        if self.model_type == "conformal":
            self.nonconformity_scores = data["nonconformity_scores"]
        self.is_trained = True
