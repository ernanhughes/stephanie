# stephanie/components/ssp/metrics/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

SSP_METRIC_VERSION = "1.0.0"

# The ONLY order used everywhere (names + 1D values aligned to this).
SSP_METRIC_ORDER: List[str] = [
    "ssp.verifier_score",   # 0..1 (↑)  from verifier/proxy; falls back to ep.reward
    "ssp.verified",         # 0/1 (↑)
    "ssp.curriculum_difficulty",  # 0..1 (↑)
    "ssp.question_len",     # 0..1 (↑)  min(1, words / max_question_words)
    "ssp.answer_len",       # 0..1 (↑)  min(1, words / max_answer_words)
    "ssp.evidence_count",   # 0..1 (↑)  min(1, docs / max_evidence)
    "ssp.solver_steps",     # 0..1 (↓)  min(1, steps / max_steps)  (direction handled by consumers if needed)
    "ssp.score",            # 0..1 (↑)  optional from ep.meta.score
    "ssp.best_score",       # 0..1 (↑)  optional from ep.meta.best_score
    "ssp.improvement",      # 0..1 (↑)  derived: rel delta best vs score (see calculator)
    "ssp.depth",            # 0..1 (↑)  min(1, depth / max_depth)
    "ssp.novelty",          # 0..1 (↑)  optional from ep.meta.novelty
]

# Optional: directions if a composite later needs "higher-is-better" view
DIRECTION_UP: Dict[str, bool] = {
    "ssp.verifier_score": True,
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
