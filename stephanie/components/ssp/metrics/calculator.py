# stephanie/components/ssp/metrics/calculator.py
from __future__ import annotations

from typing import Dict, Any
import math

from stephanie.components.ssp.metrics.registry import SSP_METRIC_ORDER, SSP_METRIC_VERSION, MetricVector
from stephanie.components.ssp.metrics.scorable import SSPScorable

class SSPMetricsCalculator:
    """
    Deterministic, versioned SSP metrics in [0,1]. Always returns the same order.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None):
        c = cfg or {}
        self.max_question_words = int(c.get("max_question_words", 128))
        self.max_answer_words   = int(c.get("max_answer_words", 128))
        self.max_evidence       = int(c.get("max_evidence", 8))
        self.max_steps          = int(c.get("max_steps", 64))
        self.max_depth          = int(c.get("max_depth", 16))

    def score(self, s: SSPScorable) -> MetricVector:
        names = list(SSP_METRIC_ORDER)
        vmap: Dict[str, float] = {}

        # 1) reward
        vmap["ssp.reward"] = _clamp01(_fallback(s.reward, 0.0))

        # 2) verified
        vmap["ssp.verified"] = 1.0 if s.verified else 0.0

        # 3) curriculum_difficulty
        vmap["ssp.curriculum_difficulty"] = _clamp01(_fallback(s.difficulty, 0.0))

        # 4) question_len
        q_words = _word_count(s.question)
        vmap["ssp.question_len"] = _clamp01(q_words / max(1, self.max_question_words))

        # 5) answer_len
        a_words = _word_count(s.predicted_answer)
        vmap["ssp.answer_len"] = _clamp01(a_words / max(1, self.max_answer_words))

        # 6) evidence_count
        evid_cnt = len(s.evidence_docs or [])
        vmap["ssp.evidence_count"] = _clamp01(evid_cnt / max(1, self.max_evidence))

        # 7) solver_steps (normalized; note: direction is DOWN in registry)
        steps = int(s.solver_steps or 0)
        vmap["ssp.solver_steps"] = _clamp01(steps / max(1, self.max_steps))

        # 8) score  (optional)
        score = s.score
        vmap["ssp.score"] = _clamp01(_fallback(score, 0.0))

        # 9) best_score (optional)
        best_score = s.best_score
        vmap["ssp.best_score"] = _clamp01(_fallback(best_score, vmap["ssp.score"]))

        # 10) improvement (derived, monotone in [0,1])
        # If both are present use relative lift; else 0
        base = vmap["ssp.score"]
        best = vmap["ssp.best_score"]
        if best >= base and best > 0:
            # relative delta wrt best, clipped to [0,1]
            #  = (best - base) / (1 - base) gives "room-to-1" lift
            denom = max(1e-6, 1.0 - base)
            vmap["ssp.improvement"] = _clamp01((best - base) / denom)
        else:
            vmap["ssp.improvement"] = 0.0

        # 11) depth
        depth = int(s.depth or 0)
        vmap["ssp.depth"] = _clamp01(depth / max(1, self.max_depth))

        # 12) novelty (optional)
        novelty = _to01((s.meta or {}).get("novelty"))
        vmap["ssp.novelty"] = _clamp01(_fallback(novelty, 0.0))

        # Assemble fixed-order values
        values = [float(vmap.get(k, 0.0)) for k in names]
        return MetricVector(
            version=SSP_METRIC_VERSION,
            names=names,
            values=values,
            vector=vmap,
        )

def _word_count(text: str) -> int:
    if not text: return 0
    return max(0, len(str(text).strip().split()))

def _clamp01(x: float) -> float:
    return 0.0 if not math.isfinite(x) else 1.0 if x > 1.0 else (0.0 if x < 0.0 else x)

def _to01(x):
    try:
        return _clamp01(float(x))
    except Exception:
        return None

def _fallback(x, default):
    return default if x is None else x
