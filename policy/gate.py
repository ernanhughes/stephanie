# src/policy/gate.py

import logging
from typing import List, Optional

import numpy as np

from policy.axes.bundle import AxisBundle
from policy.custom_types import EnergyResult, EvaluationResult, DecisionTrace
from policy.geometry.claim_evidence import ClaimEvidenceGeometry
from policy.protocols.embedder import Embedder
from policy.protocols.policy import Policy
from policy.utils.math_utils import accept_margin_ratio

logger = logging.getLogger(__name__)


class VerifiabilityGate:
    """
    policy Core Gate

    Deterministic wrapper around:
        - Embedding
        - Energy computation
        - Policy decision

    No threshold tuning happens here.
    Pure execution layer.
    """

    def __init__(
        self,
        embedder: Embedder,
        energy_computer: ClaimEvidenceGeometry,
    ):
        self.embedder = embedder
        self.energy_computer = energy_computer

    # ---------------------------------------------------------
    # Energy-only computation (used for calibration)
    # ---------------------------------------------------------

    def compute_energy(
        self,
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

        if claim_vec is None:
            claim_vec_raw = self.embedder.embed([claim])
            if claim_vec_raw.shape[0] != 1:
                raise ValueError(
                    f"Unexpected claim embedding shape: {claim_vec_raw.shape}"
                )
            claim_vec = claim_vec_raw[0]

        if evidence_vecs is None:
            evidence_vecs = self.embedder.embed(evidence_texts)

        return self.energy_computer.compute(claim_vec, evidence_vecs)

    def compute_axes(
        self,
        claim: str,
        evidence_texts: List[str],
        *,
        claim_vec: Optional[np.ndarray] = None,
        evidence_vecs: Optional[np.ndarray] = None,
    ) -> tuple[EnergyResult, AxisBundle, dict]:
        """
        Compute energy + AxisBundle once.
        Use this for policy sweeps so policies don't "soak" measurement by recomputing.
        """

        if claim_vec is None:
            claim_vec_raw = self.embedder.embed([claim])
            if claim_vec_raw.shape[0] != 1:
                raise ValueError(f"Unexpected claim embedding shape: {claim_vec_raw.shape}")
            claim_vec = claim_vec_raw[0]

        if evidence_vecs is None:
            evidence_vecs = self.embedder.embed(evidence_texts)

        base = self.energy_computer.compute(claim_vec, evidence_vecs)

        # Explained mass (preferred for "alignment" policies)
        explained = getattr(base, "explained", None)
        if explained is None:
            explained = float(1.0 - float(base.energy))

        axes = AxisBundle({
            "energy": float(base.energy),
            "explained": float(explained),
            "participation_ratio": float(base.geometry.participation_ratio),
            "sensitivity": float(base.geometry.sensitivity),
            "alignment": float(base.geometry.alignment_to_sigma1),
            "sim_margin": float(base.geometry.sim_margin),
        })

        embedding_info = {
            "claim_dim": int(np.asarray(claim_vec).shape[0]),
            "evidence_count": int(np.asarray(evidence_vecs).shape[0]),
            "embedding_backend": self.embedder.name,
        }

        return base, axes, embedding_info

    # ---------------------------------------------------------
    # Full evaluation (energy + policy decision)
    # ---------------------------------------------------------

    def evaluate(
        self,
        claim: str,
        evidence_texts: List[str],
        policy: Policy,
        *,
        run_id: str,
        split: str = "pos",
        claim_vec: Optional[np.ndarray] = None,
        evidence_vecs: Optional[np.ndarray] = None,
    ) -> EvaluationResult:

        if claim_vec is None:
            claim_vec = self.embedder.embed([claim])[0]
        if evidence_vecs is None:
            evidence_vecs = self.embedder.embed(evidence_texts)

        base, axes, embedding_info = self.compute_axes(
            claim,
            evidence_texts,
            claim_vec=claim_vec,
            evidence_vecs=evidence_vecs,
        )
        # --- Margin-based effectiveness (diagnostic only) ---

        tau = getattr(policy, "tau_accept", None)

        if tau is None:
            effectiveness = 0.0
            margin_band = None
        else:
            effectiveness = accept_margin_ratio(energy=base.energy, tau=float(tau))
            margin_band = 0.1 * float(tau)

        # --- Policy decision ---

        verdict = policy.decide(axes, effectiveness)

        logger.debug(
            "[Gate] "
            f"E={axes.get('energy'):.4f} "
            f"PR={axes.get('participation_ratio'):.4f} "
            f"S={axes.get('sensitivity'):.4f} "
            f"| tauE={policy.tau_accept:.4f} "
            f"=> {verdict.value}"
        )

        # --- Trace object (full transparency) ---

        eff_clamped = float(max(0.0, min(1.0, float(effectiveness))))
        difficulty = float(1.0 - eff_clamped)

        decision_trace = DecisionTrace(
            energy=base.energy,
            alignment=base.geometry.alignment_to_sigma1,
            participation_ratio=base.geometry.participation_ratio,
            sensitivity=base.geometry.sensitivity,
            tau_accept=policy.tau_accept,
            tau_review=policy.tau_review,
            pr_threshold=policy.pr_threshold,
            sensitivity_threshold=policy.sensitivity_threshold,
            effectiveness=effectiveness,
            difficulty=difficulty,
            margin_band=margin_band,
            policy_name=policy.name,
            hard_negative_gap=policy.hard_negative_gap,
            verdict=verdict.value,
        )

        embedding_info = {
            "claim_dim": int(claim_vec.shape[0]),
            "evidence_count": int(evidence_vecs.shape[0]),
            "embedding_backend": self.embedder.name,
        }

        return EvaluationResult(
            run_id=run_id,
            claim=claim,
            evidence=evidence_texts,
            decision_trace=decision_trace,
            embedding_info=embedding_info,
            energy_result=base,
            effectiveness=effectiveness,
            verdict=verdict,
            policy_applied=policy.name,
            split=split,
        )

