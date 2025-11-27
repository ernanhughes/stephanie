from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


@dataclass
class SelfImprovingCritic:
    """
    Thin wrapper around a tiny classifier (LogReg) that:

      - Extracts features from FrontierLens reports
      - Tracks training runs (X, y, AUC)
      - Retrains on aggregated history when recent AUC falls below a threshold

    This is the "AI" on top of FrontierLens.
    """

    max_history: int = 20
    C: float = 1.0
    max_iter: int = 1000

    model: LogisticRegression = field(init=False)
    feature_names: List[str] = field(init=False)
    history: List[Dict[str, np.ndarray]] = field(default_factory=list)
    is_trained: bool = field(default=False)

    def __post_init__(self) -> None:
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="liblinear",
            class_weight="balanced",
        )
        self.feature_names = [
            "stability",          # std of region means
            "middle_dip",         # region2 vs ends
            "global_std",         # report.global.std
            "sparsity_e3",        # report.global.sparsity_level_e3
            "entropy",            # report.global.entropy
            "trend",              # last region mean - first region mean
            "mid_bad_ratio",      # region2 / mean(region0,1,3,...) approx
            "frontier_util",      # report.global.frontier_frac
            "band_separation",    # mean(good) - mean(bad) for frontier metric
        ]

    # ----------------- Training / prediction -----------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        run_id: str,
    ) -> float:
        """
        Train on a new batch of examples and record this run.

        X: shape (N, D) features from FrontierLens (per-episode or per-cohort)
        y: shape (N,) labels (1 = good reasoning, 0 = bad)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError(f"SelfImprovingCritic.train: X must be 2D, got {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"SelfImprovingCritic.train: y shape mismatch {y.shape} vs X.shape[0] {X.shape[0]}"
            )

        self.model.fit(X, y)
        probs = self.model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)

        self.history.append(
            {
                "X": X,
                "y": y,
                "auc": np.array([auc], dtype=np.float32),
                "run_id": np.array([run_id]),
            }
        )
        if len(self.history) > self.max_history:
            self.history.pop(0)

        self.is_trained = True
        log.info(
            "SelfImprovingCritic: trained on %d examples (AUC=%.3f, run_id=%s)",
            X.shape[0],
            auc,
            run_id,
        )
        return float(auc)

    def predict(self, features: np.ndarray) -> float:
        """Predict reasoning quality (0–1) for a single feature vector."""
        if not self.is_trained:
            log.warning("SelfImprovingCritic.predict: model not trained, returning 0.5")
            return 0.5

        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(self.model.predict_proba(x)[0, 1])

    # ----------------- Feature extraction -----------------

    def extract_features(
        self,
        report_dict: Dict,
        *,
        frontier_values: Optional[np.ndarray] = None,
        frontier_low: Optional[float] = None,
        frontier_high: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extract a compact feature vector from a FrontierLens report.

        report_dict:
            FrontierLensReport.to_dict() output
        frontier_values / frontier_low / frontier_high:
            Optional; used to compute band_separation (mean(good) - mean(bad)).
        """
        regions = report_dict.get("regions", [])
        global_stats = report_dict.get("global", {})

        if len(regions) < 3:
            # pad with zeros if too few regions
            region_means = [0.0, 0.0, 0.0, 0.0]
        else:
            region_means = [float(r.get("mean_frontier_value", 0.0) or 0.0) for r in regions]

        # stability: spread across regions
        stability = float(np.std(region_means)) if region_means else 0.0

        # middle_dip: "valley" in central region vs extremes
        first = region_means[0]
        middle = region_means[len(region_means) // 2]
        last = region_means[-1]
        middle_dip = float(middle - min(first, last))

        global_std = float(global_stats.get("std", 0.0) or 0.0)
        sparsity_e3 = float(global_stats.get("sparsity_level_e3", 0.0) or 0.0)
        entropy = float(global_stats.get("entropy", 0.0) or 0.0)
        trend = float(last - first)

        # mid_bad_ratio: relative central "badness"
        avg_all = float(np.mean(region_means)) if region_means else 1.0
        mid_bad_ratio = float(middle / avg_all) if avg_all != 0.0 else 1.0

        frontier_util = float(global_stats.get("frontier_frac", 0.0) or 0.0)

        # band_separation: mean(good_band) - mean(bad_band)
        band_separation = 0.0
        if (
            frontier_values is not None
            and frontier_low is not None
            and frontier_high is not None
        ):
            vals = np.asarray(frontier_values, dtype=np.float64)
            good_mask = (vals >= frontier_low) & (vals <= frontier_high)
            bad_mask = ~good_mask

            if good_mask.any() and bad_mask.any():
                good_mean = float(vals[good_mask].mean())
                bad_mean = float(vals[bad_mask].mean())
                band_separation = good_mean - bad_mean

        feats = np.array(
            [
                stability,
                middle_dip,
                global_std,
                sparsity_e3,
                entropy,
                trend,
                mid_bad_ratio,
                frontier_util,
                band_separation,
            ],
            dtype=np.float32,
        )
        return feats

    # ----------------- History-aware retraining -----------------

    def retrain_if_needed(self, min_auc: float = 0.85) -> bool:
        """
        Retrain on combined historical data if the most recent AUC is < min_auc.

        Returns:
            True if retraining was performed, False otherwise.
        """
        if not self.is_trained or len(self.history) < 2:
            return False

        recent_auc = float(self.history[-1]["auc"][0])
        if recent_auc >= min_auc:
            return False

        # Combine historical data (more weight for recent runs)
        X_list = []
        y_list = []
        w_list = []

        n_runs = len(self.history)
        for i, h in enumerate(self.history):
            Xh = h["X"]
            yh = h["y"]
            # recency weight: linearly increasing from 0.5 to 1.5
            recency_weight = 0.5 + (i / max(1, n_runs - 1))  # in [0.5, 1.5]
            X_list.append(Xh)
            y_list.append(yh)
            w_list.append(np.full_like(yh, recency_weight, dtype=np.float32))

        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        w_all = np.concatenate(w_list)

        self.model.fit(X_all, y_all, sample_weight=w_all)
        probs = self.model.predict_proba(X_all)[:, 1]
        new_auc = roc_auc_score(y_all, probs)

        log.info(
            "SelfImprovingCritic: retrained on %d examples (AUC=%.3f, "
            "old_recent_auc=%.3f)",
            X_all.shape[0],
            new_auc,
            recent_auc,
        )
        return True
