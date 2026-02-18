# policy/core/policy_container.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional

import numpy as np

from policy.core.protocols import CalibratorProtocol


# ============================================================
# Decision Object
# ============================================================

@dataclass
class PolicyDecision:
    verdict: str
    energy: float
    margin: float
    drift_score: float
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

    Completely independent of Stephanie internals.
    """

    def __init__(
        self,
        ai_callable: Callable[..., Any],
        energy_function: Callable[[Any, Dict], float],
        calibrator: Optional[CalibratorProtocol] = None,
        review_margin: float = 0.0,
        reject_margin: float = -2.0,
        drift_threshold: float = 2.0,
    ):

        self.ai_callable = ai_callable
        self.energy_function = energy_function
        self.calibrator = calibrator

        self.review_margin = review_margin
        self.reject_margin = reject_margin
        self.drift_threshold = drift_threshold

        self.recent_energies: List[float] = []

    # ------------------------------------------------------------
    # Main Execution
    # ------------------------------------------------------------

    def execute(self, *args, context: Optional[Dict] = None, **kwargs):

        context = context or {}

        # 1️⃣ Run AI
        output = self.ai_callable(*args, **kwargs)

        # 2️⃣ Compute energy externally
        energy = float(self.energy_function(output, context))

        self.recent_energies.append(energy)

        # 3️⃣ Update calibrator (optional)
        if self.calibrator:
            self.calibrator.update(energy)

        # 4️⃣ Compute margin
        if self.calibrator:
            margin = self.calibrator.margin_score(energy)
        else:
            # raw fallback
            margin = -energy

        # 5️⃣ Drift detection
        if self.calibrator:
            drift_score = self.calibrator.detect_drift(
                self.recent_energies[-100:]
            )
        else:
            drift_score = 0.0

        # 6️⃣ Decision
        verdict = self._decide(margin)

        decision = PolicyDecision(
            verdict=verdict,
            energy=energy,
            margin=margin,
            drift_score=drift_score,
            timestamp=time.time(),
            metadata=context,
        )

        return output, decision

    # ------------------------------------------------------------
    # Decision Logic
    # ------------------------------------------------------------

    def _decide(self, margin: float) -> str:

        if margin <= self.reject_margin:
            return "REJECT"

        if margin <= self.review_margin:
            return "REVIEW"

        return "ACCEPT"

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def diagnostics(self) -> Dict[str, Any]:

        recent_mean = (
            float(np.mean(self.recent_energies[-50:]))
            if self.recent_energies else None
        )

        return {
            "recent_energy_mean": recent_mean,
            "recent_count": len(self.recent_energies),
        }
