# stephanie/components/ssp/metrics/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

SSP_METRIC_VERSION = "1.0.0"

# The ONLY order used everywhere (names + 1D values aligned to this).
# Fixed order (contract). Keep stable over time.
SSP_METRIC_ORDER: List[str] = [
    "ssp.verifier_score",
    "ssp.verified",
    "ssp.curriculum_difficulty",
    "ssp.question_len",
    "ssp.answer_len",
    "ssp.evidence_count",
    "ssp.solver_steps",        # note: direction = DOWN (fewer is better)
    "ssp.score",               # optional extras below (filled as 0.0 if missing)
    "ssp.best_score",
    "ssp.improvement",
    "ssp.depth",
    "ssp.novelty",
]

# Optional: directions if a composite later needs "higher-is-better" view
DIRECTION_UP: Dict[str, bool] = {
    "ssp.reward": True,
    "ssp.verified": True,
    "ssp.curriculum_difficulty": True,
    "ssp.question_len": True,
    "ssp.answer_len": True,
    "ssp.evidence_count": True,
    "ssp.solver_steps": False,   # ↓
    "ssp.score": True,
    "ssp.best_score": True,
    "ssp.improvement": True,
    "ssp.depth": True,
    "ssp.novelty": True,
}

@dataclass(frozen=True)
class MetricVector:
    version: str
    names: List[str]
    values: List[float]
    vector: Dict[str, float]
