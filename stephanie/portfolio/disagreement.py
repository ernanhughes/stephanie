# stephanie/portfolio/disagreement.py
"""Disagreement as a first-class object (§11–§12).

Never reduced to score variance: dimension gaps are one signal;
claim-level divergence (explicit claims + evidence) is the deeper one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence
from uuid import uuid4

from stephanie.evaluation.evaluation import Evaluation
from stephanie.portfolio.executor import PortfolioExecution

FACTUAL = "FACTUAL"
RECOMMENDATION = "RECOMMENDATION"
ASSUMPTION = "ASSUMPTION"
INTERPRETATION = "INTERPRETATION"
MISSING_INFORMATION = "MISSING_INFORMATION"
METHOD = "METHOD"
CONFIDENCE = "CONFIDENCE"


@dataclass(frozen=True)
class Disagreement:
    disagreement_id: str

    candidate_ids: tuple[str, ...]

    dimension: str

    disagreement_type: str

    severity: float | None

    description: str | None

    evidence_ids: tuple[str, ...] = ()

    metadata: Mapping[str, Any] = field(default_factory=dict)


def _sentences(text: str) -> set[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip().lower())
    return {p.strip() for p in parts if len(p.strip()) > 12}


class DisagreementAnalyzer:
    """Deterministic 3.3 extraction: dimension gaps + claim divergence."""

    def __init__(
        self,
        gap_threshold: float = 0.15,
        claim_extractor: Optional[Callable[[str], Sequence[str]]] = None,
    ):
        self.gap_threshold = gap_threshold
        self.claim_extractor = claim_extractor or (lambda text: sorted(_sentences(text)))

    def analyze(
        self,
        executions: Sequence[PortfolioExecution],
        evaluations: Sequence[Evaluation],
        scores_by_evaluation: Mapping[str, Sequence] | None = None,
    ) -> list[Disagreement]:
        disagreements: list[Disagreement] = []
        eval_by_candidate = {e.metadata.get("portfolio_candidate_id"): e for e in evaluations}
        by_id = {e.candidate_id: e for e in executions}

        # 1. Dimension gaps across evaluated candidates.
        dims: dict[str, dict[str, float]] = {}
        for execution in executions:
            evaluation = eval_by_candidate.get(execution.candidate_id)
            if evaluation is None:
                continue
            for score in (scores_by_evaluation or {}).get(evaluation.evaluation_id, []):
                dims.setdefault(score.dimension, {})[execution.candidate_id] = score.value
        for dimension, values in dims.items():
            if len(values) < 2:
                continue
            spread = max(values.values()) - min(values.values())
            if spread >= self.gap_threshold:
                ordered = sorted(values, key=values.get)
                disagreements.append(
                    Disagreement(
                        disagreement_id=f"dis_{uuid4().hex[:10]}",
                        candidate_ids=tuple(ordered),
                        dimension=dimension,
                        disagreement_type=CONFIDENCE if dimension == "confidence" else INTERPRETATION,
                        severity=min(1.0, spread),
                        description=(
                            f"score spread {spread:.2f} on '{dimension}' "
                            f"({ordered[0]} lowest, {ordered[-1]} highest)"
                        ),
                    )
                )

        # 2. Claim-level divergence: claims unique to one candidate.
        claims: dict[str, set[str]] = {
            e.candidate_id: set(self.claim_extractor(e.output_text)) for e in executions if e.success
        }
        all_claims: set[str] = set().union(*claims.values()) if claims else set()
        for candidate_id, own in claims.items():
            others = set().union(*(v for k, v in claims.items() if k != candidate_id)) if len(claims) > 1 else set()
            unique = own - others
            if unique and all_claims:
                disagreements.append(
                    Disagreement(
                        disagreement_id=f"dis_{uuid4().hex[:10]}",
                        candidate_ids=tuple(sorted(claims)),
                        dimension="claim_coverage",
                        disagreement_type=MISSING_INFORMATION,
                        severity=min(1.0, len(unique) / max(1, len(all_claims))),
                        description=f"{candidate_id} holds {len(unique)} claim(s) no other candidate states",
                        metadata={"unique_claims": sorted(unique)[:5],
                                  "candidate_id": candidate_id},
                    )
                )
        return disagreements
