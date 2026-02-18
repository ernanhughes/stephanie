# certum/protocols/geometry.py

from typing import List, Protocol, Tuple

import numpy as np

from policy.custom_types import EnergyResult


class GeometryComputer(Protocol):
    """
    Claim–Evidence geometric analysis interface.

    Responsible for:
        - Projection-based residual computation
        - Spectral diagnostics
        - Similarity ambiguity metrics
        - Sensitivity diagnostics
        - Robustness probes

    Must NOT:
        - Apply policy
        - Perform thresholding
        - Load data
        - Perform embedding

    Pure geometric computation only.
    """

    def compute(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> EnergyResult:
        """
        Compute projection energy and full geometry diagnostics.
        """
        ...

    def compute_robustness_probe(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
        param_variants: List[Tuple[int, int]],
    ) -> List[float]:
        """
        Evaluate stability of energy under parameter variation.
        """
        ...

    def compute_sensitivity(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> float:
        """
        Leave-one-out brittleness probe.
        """
        ...
