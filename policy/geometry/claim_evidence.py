import logging
from typing import List, Tuple

import numpy as np
from scipy.linalg import svd

from policy.custom_types import EnergyResult, GeometryDiagnostics

logger = logging.getLogger(__name__)


class ClaimEvidenceGeometry:
    """
    Computes hallucination energy and full geometric diagnostics
    for a claim–evidence embedding pair.

    Core outputs:
        - Energy (1 - explained variance)
        - Spectral diagnostics
        - Similarity ambiguity
        - Robustness sensitivity
        - Alignment to dominant spectral axis
    """

    def __init__(self, top_k: int = 12, rank_r: int = 8):
        self.top_k = top_k
        self.rank_r = rank_r

    # ============================================================
    # MAIN ENTRY
    # ============================================================

    def compute(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> EnergyResult:
        """
        Compute hallucination energy and geometry diagnostics.
        """

        logger.debug("Starting energy computation.")

        claim_vec, evidence_vecs = self._validate_and_prepare(
            claim_vec, evidence_vecs
        )

        if evidence_vecs.size == 0:
            logger.warning("Empty evidence vectors received.")
            return EnergyResult(
                energy=1.0,
                explained=0.0,
                identity_error=1.0,
                evidence_topk=0,
                rank_cap=0,
                geometry=None
            )

        # Normalize
        c = _unit_norm(claim_vec)
        E = _unit_norm_rows(evidence_vecs)

        # Build spectral basis
        basis, effective_rank, S, Vt = self._build_evidence_basis(c, E)

        # ========================================================
        # Spectral Metrics
        # ========================================================

        eps = 1e-12
        sum_sigma = float(np.sum(S))
        sum_sq = float(np.sum(S ** 2))

        sigma1 = float(S[0]) if len(S) > 0 else 0.0
        sigma2 = float(S[1]) if len(S) > 1 else 0.0

        sigma1_ratio = sigma1 / max(sum_sigma, eps)
        sigma2_ratio = sigma2 / max(sum_sigma, eps)

        participation_ratio = (sum_sigma ** 2) / (sum_sq + eps)

        # Entropy rank (more stable than raw rank)
        if sum_sigma > eps:
            p = S / sum_sigma
            p = p[p > eps]
            entropy_rank = float(np.exp(-np.sum(p * np.log(p))))
        else:
            entropy_rank = 0.0

        # ========================================================
        # Projection Energy
        # ========================================================

        projected = basis.T @ c if basis.shape[1] > 0 else np.zeros(0)
        explained = float(np.dot(projected, projected))
        energy = float(np.clip(1.0 - explained, 0.0, 1.0))
        identity_error = abs(1.0 - (explained + energy))

        # ========================================================
        # Similarity Metrics
        # ========================================================

        sims = E @ c
        sorted_sims = np.sort(sims)[::-1]

        sim_top1 = float(sorted_sims[0]) if len(sorted_sims) > 0 else 0.0
        sim_top2 = float(sorted_sims[1]) if len(sorted_sims) > 1 else 0.0
        sim_margin = sim_top1 - sim_top2

        # ========================================================
        # Sensitivity (concentration proxy)
        # ========================================================

        k = min(self.top_k, E.shape[0])
        idx = np.argsort(-sims)[:k]
        sims_topk = np.maximum(sims[idx], 0.0)

        if np.sum(sims_topk) < eps:
            sensitivity = 1.0
        else:
            sensitivity = float(np.max(sims_topk) / np.sum(sims_topk))

        # ========================================================
        # Alignment to dominant spectral axis
        # ========================================================

        if Vt.shape[0] > 0:
            v1 = Vt[0]
            v1 = v1 / (np.linalg.norm(v1) + eps)
            alignment = float(abs(np.dot(c, v1)))
        else:
            v1 = np.zeros((E.shape[1],), dtype=np.float32)
            alignment = 0.0

        # ========================================================
        # Geometry Object
        # ========================================================

        geometry = GeometryDiagnostics(
            sigma1_ratio=sigma1_ratio,
            sigma2_ratio=sigma2_ratio,
            spectral_sum=sum_sigma,
            participation_ratio=float(participation_ratio),
            effective_rank=int(effective_rank),
            used_count=int(E.shape[0]),
            entropy_rank=entropy_rank,
            alignment_to_sigma1=alignment,
            sim_top1=sim_top1,
            sim_top2=sim_top2,
            sim_margin=sim_margin,
            sensitivity=sensitivity,
            v1=v1.astype(np.float32),
        )

        logger.debug(
            "Energy computed: energy=%.4f explained=%.4f alignment=%.4f",
            energy, explained, alignment
        )

        return EnergyResult(
            energy=energy,
            explained=explained,
            identity_error=identity_error,
            evidence_topk=min(self.top_k, E.shape[0]),
            rank_cap=self.rank_r,
            geometry=geometry,
        )

    # ============================================================
    # ROBUSTNESS PROBE
    # ============================================================

    def compute_robustness_probe(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
        param_variants: List[Tuple[int, int]] = [(8, 6), (12, 8), (20, 10)],
    ) -> List[float]:
        """
        Evaluate energy stability under parameter variation.
        """

        logger.debug("Running robustness probe.")

        probes = []

        for top_k, rank_r in param_variants:
            try:
                computer = ClaimEvidenceGeometry(top_k=top_k, rank_r=rank_r)
                res = computer.compute(claim_vec, evidence_vecs)
                probes.append(res.energy)
            except Exception as e:
                logger.exception("Robustness probe failed: %s", str(e))
                probes.append(1.0)

        return probes

    # ============================================================
    # SENSITIVITY (LOO Spike)
    # ============================================================

    def compute_sensitivity(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> float:
        """
        Leave-one-out brittleness test.
        """

        n = len(evidence_vecs)
        if n <= 1:
            return 1.0

        base_energy = self.compute(claim_vec, evidence_vecs).energy

        max_spike = 0.0
        for i in range(n):
            loo = np.delete(evidence_vecs, i, axis=0)
            loo_energy = self.compute(claim_vec, loo).energy
            spike = loo_energy - base_energy
            max_spike = max(max_spike, spike)

        return float(max(0.0, max_spike))

    # ============================================================
    # BASIS BUILDER
    # ============================================================

    def _build_evidence_basis(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:

        if evidence_vecs.shape[0] == 0:
            return (
                np.zeros((claim_vec.shape[0], 0), dtype=np.float32),
                0,
                np.array([]),
                np.array([]),
            )

        E_topk = evidence_vecs

        try:
            _, S, Vt = svd(E_topk, full_matrices=False)
        except np.linalg.LinAlgError:
            logger.exception("SVD failed.")
            d = evidence_vecs.shape[1]
            return (
                np.zeros((d, 0), dtype=np.float32),
                0,
                np.array([]),
                np.array([]),
            )

        r = min(self.rank_r, Vt.shape[0])
        basis = Vt[:r].T
        effective_rank = int(np.sum(S > 1e-6))

        return basis.astype(np.float32), effective_rank, S, Vt

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_and_prepare(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ):
        claim_vec = np.asarray(claim_vec, dtype=np.float32)
        evidence_vecs = np.asarray(evidence_vecs, dtype=np.float32)

        if claim_vec.ndim != 1:
            raise ValueError(f"claim_vec must be 1D vector, got {claim_vec.shape}")

        if evidence_vecs.ndim == 1:
            evidence_vecs = evidence_vecs.reshape(1, -1)
        elif evidence_vecs.ndim != 2:
            raise ValueError(f"evidence_vecs must be 2D, got {evidence_vecs.shape}")

        if claim_vec.shape[0] != evidence_vecs.shape[1]:
            raise ValueError("Dimension mismatch claim/evidence")

        return claim_vec, evidence_vecs


# ================================================================
# Normalization Utilities
# ================================================================

def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    return x / max(norm, eps)


def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms
