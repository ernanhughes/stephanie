from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class Calibrator(Protocol):
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
