# stephanie/components/ssp/metrics/scorable.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Minimal shape we need from the solver/tree. Keep it decoupled.
@dataclass
class SSPScorable:
    episode_id: str
    question: str
    predicted_answer: str
    verified: bool
    seed_answer: Optional[str] = None
    reward: Optional[float] = None  # 0..1 if available
    difficulty: Optional[float] = None  # 0..1 if available
    solver_steps: Optional[int] = None
    evidence_docs: Optional[List[Any]] = None
    depth: Optional[int] = None
    meta: Optional[Dict[str, Any]] = (
        None  # may include score, best_score, novelty
    )

    @staticmethod
    def from_episode_trace(ep) -> "SSPScorable":
        # ep is your existing EpisodeTrace
        return SSPScorable(
            episode_id=str(getattr(ep, "episode_id", "")),
            question=getattr(ep, "question", "") or "",
            predicted_answer=getattr(ep, "predicted_answer", "") or "",
            seed_answer=str(getattr(ep, "seed_answer", "")),
            verified=bool(getattr(ep, "verified", False)),
            reward=_to01(getattr(ep, "reward", None)),
            difficulty=_to01(getattr(ep, "difficulty", None)),
            solver_steps=int(getattr(ep, "solver_steps", 0) or 0),
            evidence_docs=list(getattr(ep, "evidence_docs", []) or []),
            depth=int((getattr(ep, "meta", {}) or {}).get("depth", 0) or 0),
            meta=getattr(ep, "meta", {}) or {},
        )


def _to01(x):
    try:
        v = float(x)
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v
    except Exception:
        return None
