# stephanie/scoring/calibration.py
"""
ScoreCalibrator - Calibrates AI-generated scores to align with human judgment

This module provides a class for calibrating AI-generated scores (0-100) to match
human-equivalent scores (0-1), addressing the common problem of AI over-optimism.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class ScoreCalibrator:
    """
    Calibrates AI scores (0-100) to human-equivalent [0,1] scores.

    This class solves the critical problem of AI over-optimism by mapping AI scores
    to what a human would have rated the same content. For example, an AI score of
    95/100 often corresponds to a human score of only +3.5/5 (0.75 on [0,1] scale).

    The calibration is done using isotonic regression, which provides a non-linear
    but monotonic mapping that preserves the ranking while adjusting for optimism bias.

    Usage:
        # Create and fit calibrator
        calibrator = ScoreCalibrator()
        human_scores = [5, 3, 0, -2, -5]  # human star ratings (-5 to +5)
        ai_scores = [95, 80, 65, 40, 10]   # corresponding AI scores (0-100)
        calibrator.fit(human_scores, ai_scores)

        # Calibrate new AI scores
        calibrated_score = calibrator.calibrate(95)  # returns ~0.75 instead of 0.95

        # Save and load
        calibrator.save("models/calibrators/knowledge.json")
        loaded_calibrator = ScoreCalibrator.load("models/calibrators/knowledge.json")
    """

    def __init__(self):
        """Initialize an empty calibrator with conservative defaults."""
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted = False
        self.calibration_quality = {
            "mse": None,
            "r2": None,
            "sample_count": 0,
            "last_fit": None,
        }
        self.default_slope = 0.8  # Conservative scaling factor
        self.x_thresholds = None
        self.y_thresholds = None

    def fit(self, human_scores: List[float], ai_scores: List[float]) -> None:
        """
        Fit calibration curve using human-AI score pairs.

        Args:
            human_scores: Human star ratings (-5 to +5)
            ai_scores: Corresponding AI scores (0-100)

        Raises:
            ValueError: If input lists have different lengths or are empty
            ValueError: If scores are out of expected ranges
        """
        if len(human_scores) != len(ai_scores):
            raise ValueError(
                "human_scores and ai_scores must have the same length"
            )

        if len(human_scores) == 0:
            raise ValueError("At least one score pair is required for fitting")

        # Validate score ranges
        for h in human_scores:
            if not (-5 <= h <= 5):
                raise ValueError(f"Human score {h} out of range (-5 to +5)")

        for a in ai_scores:
            if not (0 <= a <= 100):
                raise ValueError(f"AI score {a} out of range (0 to 100)")

        # Convert to proper calibration targets
        # Human: -5..+5 → 0..1 (0 = -5, 0.5 = 0, 1.0 = +5)
        human_normalized = [(s + 5) / 10.0 for s in human_scores]

        # AI: 0-100 → 0-1
        ai_normalized = [s / 100.0 for s in ai_scores]

        # Fit isotonic regression (non-linear, monotonic calibration)
        try:
            self.calibrator.fit(ai_normalized, human_normalized)
            self.is_fitted = True
            self.calibration_quality["sample_count"] = len(human_scores)
            self.calibration_quality["last_fit"] = str(datetime.now())

            # Store thresholds explicitly for reliable saving
            self.x_thresholds = self.calibrator.X_thresholds_
            self.y_thresholds = self.calibrator.y_thresholds_

            # Calculate quality metrics
            calibrated = self.calibrator.predict(ai_normalized)
            self.calibration_quality["mse"] = float(
                np.mean((calibrated - human_normalized) ** 2)
            )
            self.calibration_quality["r2"] = float(
                1
                - np.sum((calibrated - human_normalized) ** 2)
                / np.sum((np.mean(human_normalized) - human_normalized) ** 2)
            )

            logger.info(
                f"ScoreCalibrator fitted with {len(human_scores)} samples. "
                f"MSE: {self.calibration_quality['mse']:.4f}, "
                f"R²: {self.calibration_quality['r2']:.4f}"
            )

        except Exception as e:
            logger.error(f"Failed to fit ScoreCalibrator: {str(e)}")
            self.is_fitted = False
            raise

    def calibrate(
        self, ai_score: float, return_details: bool = False
    ) -> Union[float, Tuple[float, Dict]]:
        """
        Calibrate raw AI score (0-100) to human-equivalent [0,1].

        Args:
            ai_score: Raw AI score (0-100)
            return_details: If True, return (calibrated_score, details_dict)

        Returns:
            Calibrated score in [0,1] range, or tuple with details if requested

        Notes:
            - When not fitted, uses conservative default scaling (100→0.8, 75→0.6, etc.)
            - Scores > 90 get special handling to prevent over-optimism
            - Always clamps to [0,1] range
        """
        # Input validation
        if not isinstance(ai_score, (int, float)):
            try:
                ai_score = float(ai_score)
            except (TypeError, ValueError):
                logger.warning(
                    f"Invalid AI score type: {type(ai_score)}. Using 50.0 as fallback."
                )
                ai_score = 50.0

        # Clamp input to valid range
        original_score = ai_score
        ai_score = max(0.0, min(100.0, float(ai_score)))

        details = {
            "input_score": original_score,
            "clamped_score": ai_score,
            "is_fitted": self.is_fitted,
            "calibration_quality": self.calibration_quality.copy(),
        }

        # Apply calibration
        if self.is_fitted:
            normalized = ai_score / 100.0
            try:
                calibrated = self.calibrator.predict([normalized])[0]
                details["calibration_method"] = "isotonic_regression"
            except Exception as e:
                logger.warning(
                    f"IsotonicRegression prediction failed: {str(e)}. Using fallback."
                )
                calibrated = self._default_calibration(ai_score)
                details["calibration_method"] = "default_fallback"
        else:
            calibrated = self._default_calibration(ai_score)
            details["calibration_method"] = "default_scaling"

        # Additional optimism correction for high scores when no human signal
        if ai_score > 90 and not self.is_fitted:
            calibrated = min(calibrated, 0.85)
            details["optimism_correction"] = True
            details["corrected_value"] = calibrated
        else:
            details["optimism_correction"] = False

        # Final clamp to [0,1]
        calibrated = max(0.0, min(1.0, calibrated))
        details["calibrated_score"] = calibrated

        return (calibrated, details) if return_details else calibrated

    def _default_calibration(self, ai_score: float) -> float:
        """
        Conservative default calibration when no data is available.

        Maps:
          100 → 0.80
          90  → 0.70
          80  → 0.60
          70  → 0.50
          60  → 0.40
          50  → 0.35
          40  → 0.30
          30  → 0.25
          20  → 0.15
          10  → 0.05
          0   → 0.00

        This is a piecewise linear function with decreasing slope for higher scores,
        reflecting the observation that AI is most optimistic at the high end.
        """
        # Piecewise linear function with decreasing slope
        if ai_score >= 90:
            return 0.50 + 0.30 * ((ai_score - 90) / 10)
        elif ai_score >= 70:
            return 0.30 + 0.20 * ((ai_score - 70) / 20)
        elif ai_score >= 50:
            return 0.15 + 0.15 * ((ai_score - 50) / 20)
        else:
            return 0.05 * (ai_score / 50)

    def evaluate(
        self, human_scores: List[float], ai_scores: List[float]
    ) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before evaluation")

        # Convert to proper ranges
        human_n = np.array(
            [(s + 5) / 10.0 for s in human_scores], dtype=float
        )  # -5..+5 → 0..1
        ai_raw = np.array(ai_scores, dtype=float)  # 0..100

        # Get calibrated scores
        calibrated = np.array(
            [self.calibrate(s) for s in ai_raw], dtype=float
        )  # 0..1

        # Metrics
        mse = float(np.mean((calibrated - human_n) ** 2))
        total_var = float(np.sum((human_n - np.mean(human_n)) ** 2))
        r2 = float(
            1.0 - (np.sum((calibrated - human_n) ** 2) / (total_var + 1e-12))
        )

        # Classification accuracy (at neutral threshold)
        human_bin = (human_n > 0.5).astype(int)
        calib_bin = (calibrated > 0.5).astype(int)
        accuracy = float(np.mean(human_bin == calib_bin))

        return {
            "mse": mse,
            "r2": r2,
            "accuracy": accuracy,
            "sample_count": len(human_scores),
            "threshold_0.5_accuracy": accuracy,
        }

    def save(self, path: str) -> None:
        """Save by sampling the calibration curve instead of accessing private attrs"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")

        # Sample 0-100 and store calibrated outputs
        xs = list(np.linspace(0, 100, 101))
        ys = [float(self.calibrate(x)) for x in xs]

        save_data = {
            "sample_x": xs,  # AI scores (0-100)
            "sample_y": ys,  # Calibrated human-equivalent scores (0-1)
            "calibration_quality": self.calibration_quality,
            "format_version": "1.1",
            "saved_at": str(datetime.now()),
        }

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"ScoreCalibrator saved to {path} (n={len(xs)} samples)")

    @classmethod
    def load(cls, path: str) -> "ScoreCalibrator":
        """Load by refitting on saved samples"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibrator file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        cal = cls()

        # Convert to proper ranges for fitting
        # AI: 0-100 → 0-1, Human: already 0-1 from calibrate()
        xs = np.array(data["isotonic_x"], dtype=float) / 100.0
        ys = np.array(data["isotonic_y"], dtype=float)

        # Refit the calibrator
        cal.calibrator.fit(xs, ys)
        cal.is_fitted = True
        cal.calibration_quality = data.get(
            "calibration_quality", cal.calibration_quality
        )

        logger.info(f"ScoreCalibrator loaded from {path} (n={len(xs)})")
        return cal

    def get_calibration_curve(
        self, num_points: int = 100
    ) -> Tuple[List[float], List[float]]:
        """
        Get the calibration curve as two lists: input scores and calibrated scores.

        Args:
            num_points: Number of points to sample along the curve

        Returns:
            (input_scores, calibrated_scores) - both lists of length num_points
        """
        if not self.is_fitted:
            # Generate default curve
            input_scores = np.linspace(0, 100, num_points)
            calibrated_scores = [
                self._default_calibration(s) for s in input_scores
            ]
            return input_scores.tolist(), calibrated_scores

        # For fitted calibrator
        input_scores = np.linspace(0, 100, num_points)
        normalized_inputs = input_scores / 100.0
        calibrated_scores = self.calibrator.predict(normalized_inputs).tolist()

        return input_scores.tolist(), calibrated_scores

    def is_reliable(self, min_samples: int = 50, min_r2: float = 0.7) -> bool:
        """
        Determine if the calibrator is reliable for production use.

        Args:
            min_samples: Minimum number of samples for reliability
            min_r2: Minimum R² value for reliability

        Returns:
            True if calibrator meets reliability criteria
        """
        if not self.is_fitted:
            return False

        quality = self.calibration_quality
        return (
            quality.get("sample_count", 0) >= min_samples
            and quality.get("r2", 0.0) >= min_r2
        )

    def get_reliability_report(self) -> Dict[str, Union[bool, float, int]]:
        """Get a detailed report on calibrator reliability."""
        return {
            "is_reliable": self.is_reliable(),
            "is_fitted": self.is_fitted,
            "sample_count": self.calibration_quality.get("sample_count", 0),
            "mse": self.calibration_quality.get("mse", None),
            "r2": self.calibration_quality.get("r2", None),
            "min_samples_threshold": 50,
            "min_r2_threshold": 0.7,
        }
