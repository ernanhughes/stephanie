# stephanie/components/ssp/metrics/scorable.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from stephanie.components.ssp.utils.trace import EpisodeTrace


# Minimal shape we need from the solver/tree. Keep it decoupled.
@dataclass
class SSPScorable:
    episode_id: str
    question: str
    predicted_answer: str
    verified: bool
    best_score: Optional[float] = None
    score: Optional[float] = None
    seed_answer: Optional[str] = None
    reward: Optional[float] = None  # 0..1 if available
    difficulty: Optional[float] = None  # 0..1 if available
    solver_steps: Optional[int] = None
    evidence_docs: Optional[List[Any]] = None
    depth: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None  # may include score, best_score, novelty
    

    @staticmethod
    def from_episode_trace(ep: EpisodeTrace) -> SSPScorable:
        # ep is your existing EpisodeTrace
        depth = ep.solver_meta.get("depth", 0) if ep.solver_meta else 0
        best_score = ep.solver_meta.get("best_score", 0) if ep.solver_meta else 0
        reward = ep.reward/100 if ep.reward is not None else 0.0
        difficulty = ep.difficulty if ep.difficulty is not None else 0.0
        return SSPScorable(
            episode_id=ep.episode_id,
            question=ep.question,
            predicted_answer=ep.predicted_answer,
            seed_answer=ep.seed_answer,
            verified=ep.verified,
            reward=reward,
            best_score=best_score,
            score=best_score,
            difficulty=difficulty,
            solver_steps=ep.solver_steps,
            evidence_docs=ep.evidence_docs,
            depth=depth,
            meta=ep.to_dict(),
        )
