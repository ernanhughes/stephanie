from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class Calibrator(Protocol):

    from typing import Protocol, List, Optional


class CalibratorProtocol(Protocol):

    def update(self, energy: float) -> None:
        ...

    def margin_score(self, energy: float) -> float:
        ...

    def detect_drift(self, recent_history: List[float]) -> float:
        ...

    def recalibrate(
        self,
        positive_energies: List[float],
        hard_negative_energies: Optional[List[float]] = None,
    ) -> None:
        ...

    def diagnostics(self) -> dict:
        ...

    """
    Threshold calibration interface.

    Responsible for:
        - Computing energy-based thresholds
        - Enforcing target FAR
        - Producing calibration statistics

    Must NOT:
        - Load datasets
        - Apply final policy
        - Write files

    Pure calibration logic only.
    """

    def run_sweep(
        self,
        *,
        claims: List[str],
        evidence_sets: List[List[str]],
        evidence_vecs: List[np.ndarray],
        percentiles: List[int],
        neg_mode: str,
        seed: int,
        neg_offset: Optional[int] = None,
        claim_vec_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Executes calibration sweep.

        Returns:
            {
                "tau_energy": float,
                "tau_pr": float,
                "tau_sensitivity": float,
                "hard_negative_gap": float,
                ...
            }
        """
        ...
