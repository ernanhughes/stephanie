# policy/calibration/adaptive_calibrator.py

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CalibrationResult:
    tau_energy: float
    energy_mean: float
    energy_std: float
    hard_negative_gap: float
    hard_negative_gap_norm: float
    positive_count: int
    negative_count: int


class AdaptiveCalibrator:
    """
    Distribution-aware energy calibration.

    Learns tau_energy from positive distribution and optionally
    incorporates hard-negative gap normalization.

    Completely model-agnostic.
    """

    def __init__(
        self,
        percentile: float = 95.0,
        min_samples: int = 50,
        eps: float = 1e-8,
    ):
        """
        Args:
            percentile:
                Percentile of positive energy distribution to use as τ.
            min_samples:
                Minimum required samples before trusting calibration.
            eps:
                Numerical stability constant.
        """
        self.percentile = percentile
        self.min_samples = min_samples
        self.eps = eps

    # ------------------------------------------------------------
    # Primary Calibration
    # ------------------------------------------------------------

    def calibrate(
        self,
        positive_energies: List[float],
        hard_negative_energies: Optional[List[float]] = None,
    ) -> CalibrationResult:
        """
        Calibrate energy threshold using positive and hard-negative distributions.

        Args:
            positive_energies:
                Energy values for verified supported samples.
            hard_negative_energies:
                Energy values for adversarial / unsupported samples.

        Returns:
            CalibrationResult
        """

        pos = np.asarray(positive_energies, dtype=np.float64)

        if len(pos) < self.min_samples:
            raise ValueError(
                f"Insufficient positive samples for calibration "
                f"(required={self.min_samples}, got={len(pos)})"
            )

        energy_mean = float(np.mean(pos))
        energy_std = float(np.std(pos) + self.eps)

        tau_energy = float(np.percentile(pos, self.percentile))

        # --------------------------------------------------------
        # Hard Negative Gap
        # --------------------------------------------------------

        if hard_negative_energies:
            neg = np.asarray(hard_negative_energies, dtype=np.float64)
            gap = float(np.mean(neg) - np.mean(pos))
            gap_norm = float(gap / energy_std)
            negative_count = len(neg)
        else:
            gap = 0.0
            gap_norm = 0.0
            negative_count = 0

        return CalibrationResult(
            tau_energy=tau_energy,
            energy_mean=energy_mean,
            energy_std=energy_std,
            hard_negative_gap=gap,
            hard_negative_gap_norm=gap_norm,
            positive_count=len(pos),
            negative_count=negative_count,
        )

    # ------------------------------------------------------------
    # Margin Scoring (for runtime decisions)
    # ------------------------------------------------------------

    def margin_score(
        self,
        energy: float,
        calibration: CalibrationResult,
    ) -> float:
        """
        Compute normalized margin score relative to calibrated τ.

        Positive = safe
        Negative = risky
        """

        z = (calibration.tau_energy - energy) / (
            calibration.energy_std + self.eps
        )
        return float(z)

    # ------------------------------------------------------------
    # Drift Detection
    # ------------------------------------------------------------

    def detect_drift(
        self,
        recent_energies: List[float],
        calibration: CalibrationResult,
        drift_threshold: float = 2.0,
    ) -> bool:
        """
        Detect distribution shift via mean shift in Z-space.
        """

        recent = np.asarray(recent_energies, dtype=np.float64)

        if len(recent) == 0:
            return False

        mean_recent = float(np.mean(recent))

        z_shift = abs(
            (mean_recent - calibration.energy_mean)
            / (calibration.energy_std + self.eps)
        )

        return z_shift > drift_threshold
