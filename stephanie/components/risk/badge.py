# stephanie/components/hallucinations/badge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HallucinationBadge:
    level: str   # "OK" | "WARN" | "RISK"
    score: float
    reasons: List[str]

# stephanie/hallucinations/badge.py
def make_badge(
    se_mean: float | None,
    meta_viols: float | None,
    rag_unsupported: float | None,
) -> HallucinationBadge:
    # normalize / default
    s = max(0.0, min(1.0, (se_mean or 0.0) / 1.5))          # SE ~ 0..1.5 typical
    m = max(0.0, min(1.0, meta_viols or 0.0))               # already 0..1
    r = max(0.0, min(1.0, rag_unsupported or 0.0))          # 0..1

    # weighted blend
    score = 0.45 * s + 0.35 * r + 0.20 * m

    if score >= 0.6: level = "risk"
    elif score >= 0.35: level = "warn"
    else: level = "ok"

    return HallucinationBadge(level=level, score=score, reasons={"se": s, "meta": m, "rag": r})