from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class Verdict(Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"

@dataclass(frozen=True)
class SupportDiagnostics:
    """
    Sentence-level support analysis for summarization-style tasks.
    Aggregated over all sentences in a summary.
    """

    sentence_count: int
    paragraph_count: int

    # Entailment-style support scores
    max_entailment: float
    mean_entailment: float
    min_entailment: float

    # Similarity-based fallback metrics
    mean_sim_top1: float
    min_sim_top1: float
    mean_sim_margin: float
    min_sim_margin: float

    # Coverage signals
    mean_coverage: float
    min_coverage: float

    # Energy aggregates
    max_energy: float
    mean_energy: float
    min_energy: float
    high_energy_count: int

    p90_energy: float
    frac_above_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": {
                "sentence_count": self.sentence_count,
                "paragraph_count": self.paragraph_count,
            },
            "entailment": {
                "max": self.max_entailment,
                "mean": self.mean_entailment,
                "min": self.min_entailment,
            },
            "similarity": {
                "mean_sim_top1": self.mean_sim_top1,
                "min_sim_top1": self.min_sim_top1,
                "mean_sim_margin": self.mean_sim_margin,
                "min_sim_margin": self.min_sim_margin,
            },
            "coverage": {
                "mean": self.mean_coverage,
                "min": self.min_coverage,
            },
            "energy": {
                "max": self.max_energy,
                "min": self.min_energy,
                "high_count": self.high_energy_count,
                "mean": self.mean_energy,
                "p90": self.p90_energy,
                "frac_above_threshold": self.frac_above_threshold,
            }
        }


@dataclass
class GovernanceVector:
    """
    Normalized governance metrics in [-inf, +inf] space.

    All values must be direction-corrected:
        positive = improvement
        negative = risk
    """

    axes: Dict[str, float]

    def get(self, key: str, default: float = 0.0) -> float:
        return self.axes.get(key, default)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.axes)


@dataclass(frozen=True)
class GeometryDiagnostics:
    """
    Intrinsic geometric properties of claim–evidence interaction.
    All values are computed at SVD time.
    """

    # Spectral structure
    sigma1_ratio: float
    sigma2_ratio: float
    spectral_sum: float
    participation_ratio: float

    effective_rank: int
    used_count: int

    # Alignment
    alignment_to_sigma1: float

    # Similarity geometry
    sim_top1: float
    sim_top2: float
    sim_margin: float

    # Concentration / brittleness
    sensitivity: float

    # Optional raw vector (for offline research only)
    v1: np.ndarray
    entropy_rank: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spectral": {
                "sigma1_ratio": self.sigma1_ratio,
                "sigma2_ratio": self.sigma2_ratio,
                "spectral_sum": self.spectral_sum,
                "participation_ratio": self.participation_ratio,
                "effective_rank": self.effective_rank,
            },
            "alignment": {
                "alignment_to_sigma1": self.alignment_to_sigma1,
            },
            "similarity": {
                "sim_top1": self.sim_top1,
                "sim_top2": self.sim_top2,
                "sim_margin": self.sim_margin,
            },
            "robustness": {
                "sensitivity": self.sensitivity,
            },
            "support": {
                "effective_rank": self.effective_rank,
                "used_count": self.used_count,
                "entropy_rank": self.entropy_rank,
            },
        }


@dataclass(frozen=True)
class EnergyResult:
    energy: float
    explained: float
    identity_error: float

    evidence_topk: int
    rank_cap: int

    geometry: GeometryDiagnostics

    def is_stable(self, threshold: float = 1e-4) -> bool:
        return self.identity_error < threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.energy,
            "explained": self.explained,
            "identity_error": self.identity_error,
            "config": {
                "evidence_topk": self.evidence_topk,
                "rank_cap": self.rank_cap,
            },
            "geometry": self.geometry.to_dict(),
        }



@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation outcome."""

    claim: str
    evidence: List[str]

    energy_result: EnergyResult
    decision_trace: DecisionTrace
    verdict: Verdict
    policy_applied: str

    run_id: str
    split: str
    effectiveness: float

    embedding_info: Dict

    robustness_probe: Optional[List[float]] = None  # Energy under param variations
    difficulty_value: Optional[float] = 0.0
    difficulty_bucket: Optional[str] = None
    support_diagnostics: Optional[SupportDiagnostics] = None
    label: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "meta": {
                "run_id": self.run_id,
                "split": self.split,
                "policy": self.policy_applied,
            },
            "claim": self.claim,
            "evidence": self.evidence,
            "difficulty": {
                "value": self.difficulty_value,
                "bucket": self.difficulty_bucket,
            },
            "effectiveness": self.effectiveness,
            "embedding": self.embedding_info,
            "decision": {
                "verdict": self.verdict.value if self.verdict else None,
            },
            "stability": {
                "probe_variance": float(np.var(self.robustness_probe))
                if self.robustness_probe
                else None,
            },
            "label": self.label
        }
        

        if self.energy_result is not None:
            data["energy"] = self.energy_result.to_dict()
            data["is_stable"] = self.energy_result.is_stable() 

        if self.support_diagnostics is not None:
            data["support"] = self.support_diagnostics.to_dict()

        if self.decision_trace is not None:
            data["decision"] = {
                "verdict": self.verdict.value if self.verdict else None,
            }

        return data

@dataclass(frozen=True)
class DecisionTrace:
    """
    Deterministic explanation of a 3-axis geometry-aware policy decision.
    """

    # === Core Energy Axis ===
    energy: float
    alignment: float  # |dot(claim, v1)|

    # === Geometry Axis ===
    participation_ratio: float
    sensitivity: float
    effectiveness: float
    difficulty: float

    # Policy thresholds
    tau_accept: Optional[float]
    tau_review: Optional[float]
    pr_threshold: Optional[float]
    sensitivity_threshold: Optional[float]
    margin_band: Optional[float]

    # Policy metadata
    policy_name: str
    hard_negative_gap: float

    # Final action
    verdict: str  # expected: "accept" | "review" | "reject"

    def to_dict(self) -> dict:
        return asdict(self)


def why_rejected(trace: DecisionTrace) -> str:
    """
    Deterministic explanation for REJECT verdicts.
    """
    if trace.verdict != "reject":
        return "Not rejected."

    reasons = []

    # Energy hard-reject (most common)
    if trace.tau_review is not None and trace.energy > trace.tau_review:
        reasons.append("Energy exceeds review threshold.")

    # If you ever make PR/Sensitivity hard-reject in policy, these become exact.
    # Otherwise they are informational: they explain why the sample is risky.
    if trace.pr_threshold is not None and trace.participation_ratio > trace.pr_threshold:
        reasons.append("High PR (diffuse evidence manifold).")

    if trace.sensitivity_threshold is not None and trace.sensitivity > trace.sensitivity_threshold:
        reasons.append("High sensitivity (brittle evidence dependence).")

    if trace.effectiveness < 0.05:
        reasons.append("Insufficient effectiveness margin.")

    if not reasons:
        reasons.append("Rejected by policy fallback.")

    return " | ".join(reasons)


def why_reviewed(trace: DecisionTrace) -> str:
    """
    Deterministic explanation for REVIEW verdicts.
    """
    if trace.verdict != "review":
        return "Not reviewed."

    reasons = []

    if (
        trace.margin_band is not None
        and trace.tau_accept is not None
        and abs(trace.energy - trace.tau_accept) <= trace.margin_band
    ):
        reasons.append("Within policy margin band.")

    # These are “human-facing” interpretations; keep them stable.
    if trace.difficulty > 0.4:
        reasons.append("Moderate difficulty region.")

    if trace.effectiveness < 0.25:
        reasons.append("Low effectiveness margin.")

    # Optional: geometry triggers (only if policy actually uses these to trigger REVIEW)
    if trace.pr_threshold is not None and trace.participation_ratio > trace.pr_threshold:
        reasons.append("PR exceeds threshold.")

    if trace.sensitivity_threshold is not None and trace.sensitivity > trace.sensitivity_threshold:
        reasons.append("Sensitivity exceeds threshold.")

    if not reasons:
        reasons.append("Reviewed by policy fallback.")

    return " | ".join(reasons)
