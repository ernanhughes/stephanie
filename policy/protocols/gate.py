from typing import List, Optional, Protocol

import numpy as np

from policy.custom_types import EnergyResult, EvaluationResult
from policy.geometry.claim_evidence import ClaimEvidenceGeometry
from policy.protocols.embedder import Embedder


class Gate(Protocol):
    """
    Deterministic policy execution interface.

    A Gate implementation must:
        - Accept precomputed diagnostics
        - Apply a policy
        - Return EvaluationResult

    It must NOT:
        - Perform embedding
        - Compute energy
        - Perform calibration
        - Tune thresholds
    """

    def compute_energy(
        claim: str,
        evidence_texts: List[str],
        *,
        claim_vec: Optional[np.ndarray] = None,
        evidence_vecs: Optional[np.ndarray] = None,
    ) -> EnergyResult:
        """
        Compute hallucination energy WITHOUT policy decision.
        Used during calibration sweeps.
        """
        ...

    def evaluate(self, embedder: Embedder,
        energy_computer: ClaimEvidenceGeometry,) -> EvaluationResult:
        """
        Executes deterministic policy decision.

        Expected inputs:
            {
                "run_id": str,
                "claim": str,
                "evidence": List[str],
                "axes": AxisBundle,
                "energy_result": EnergyResult,
                "effectiveness": float,
                "embedding_info": dict,
                "split": str,
            }

        Returns:
            EvaluationResult
        """
        ...
