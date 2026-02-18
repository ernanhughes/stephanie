# policy/core/policy_container.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional

import numpy as np

from policy.calibration.adaptive_calibrator import (
    AdaptiveCalibrator,
    CalibrationResult,
)


# ============================================================
# Decision Object
# ============================================================

@dataclass
class PolicyDecision:
    verdict: str              # "ACCEPT", "REVIEW", "REJECT"
    energy: float
    margin: float
    drift_detected: bool
    timestamp: float
    metadata: Dict[str, Any]


# ============================================================
# Policy Container
# ============================================================

class PolicyContainer:
    """
    AI Governance Wrapper.

    Wraps ANY AI system and enforces calibrated
    hallucination-energy-based governance.

    Completely independent of Stephanie.
    """

    def __init__(
        self,
        ai_callable: Callable[..., Any],
        energy_function: Callable[[Any, Dict], float],
        calibrator: AdaptiveCalibrator,
        calibration: CalibrationResult,
        review_margin: float = 0.0,
        reject_margin: float = -2.0,
        drift_threshold: float = 2.0,
    ):
        """
        Args:
            ai_callable:
                Black-box AI system (e.g. Stephanie.run)

            energy_function:
                Callable that computes energy from AI output + context

            calibrator:
                AdaptiveCalibrator instance

            calibration:
                Precomputed calibration result

            review_margin:
                Z-score margin below which we go to REVIEW

            reject_margin:
                Z-score margin below which we REJECT

            drift_threshold:
                Z-score shift threshold for drift detection
        """

        self.ai_callable = ai_callable
        self.energy_function = energy_function
        self.calibrator = calibrator
        self.calibration = calibration

        self.review_margin = review_margin
        self.reject_margin = reject_margin
        self.drift_threshold = drift_threshold

        self.recent_energies: List[float] = []

    # ------------------------------------------------------------
    # Main Execution
    # ------------------------------------------------------------

    def execute(self, *args, context: Optional[Dict] = None, **kwargs) -> tuple[Any, PolicyDecision]:
        """
        Execute AI system under policy governance.
        """

        context = context or {}

        # 1️⃣ Run AI
        output = self.ai_callable(*args, **kwargs)

        # 2️⃣ Compute energy externally
        energy = float(self.energy_function(output, context))

        self.recent_energies.append(energy)

        # 3️⃣ Compute normalized margin
        margin = self.calibrator.margin_score(
            energy=energy,
            calibration=self.calibration,
        )

        # 4️⃣ Drift detection
        drift = self.calibrator.detect_drift(
            self.recent_energies[-100:],  # rolling window
            self.calibration,
            drift_threshold=self.drift_threshold,
        )

        # 5️⃣ Decision logic
        verdict = self._decide(margin)

        decision = PolicyDecision(
            verdict=verdict,
            energy=energy,
            margin=margin,
            drift_detected=drift,
            timestamp=time.time(),
            metadata=context,
        )

        return output, decision

    # ------------------------------------------------------------
    # Decision Logic
    # ------------------------------------------------------------

    def _decide(self, margin: float) -> str:
        """
        Decision boundary in normalized Z-space.
        """

        if margin <= self.reject_margin:
            return "REJECT"

        if margin <= self.review_margin:
            return "REVIEW"

        return "ACCEPT"

    # ------------------------------------------------------------
    # Recalibration Hook
    # ------------------------------------------------------------

    def recalibrate(
        self,
        positive_energies: List[float],
        hard_negative_energies: Optional[List[float]] = None,
    ):
        """
        Update calibration live.
        """

        self.calibration = self.calibrator.calibrate(
            positive_energies,
            hard_negative_energies,
        )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def diagnostics(self) -> Dict[str, Any]:
        """
        Return current policy state.
        """

        return {
            "tau_energy": self.calibration.tau_energy,
            "energy_mean": self.calibration.energy_mean,
            "energy_std": self.calibration.energy_std,
            "hard_negative_gap_norm": self.calibration.hard_negative_gap_norm,
            "recent_energy_mean": float(np.mean(self.recent_energies[-50:])) if self.recent_energies else None,
        }
