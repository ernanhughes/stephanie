"""
Episode Trace Data Structure

This module defines the EpisodeTrace class that captures the
complete state of an SSP episode, including all critical
information needed for self-play rewards and VPM visualization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass
class EpisodeTrace:
    """Complete trace of an SSP episode for analysis and visualization."""
    episode_id: str
    seed_answer: str
    question: str
    proposer_evidence: List[str]
    predicted_answer: str
    evidence_docs: List[str]
    verified: bool
    verifier_score: float
    solver_steps: int
    difficulty: float = 0.0
    proposer_meta: Dict[str, Any] = field(default_factory=dict)
    verifier_meta: Dict[str, Any] = field(default_factory=dict)
    solver_meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    episode_duration: float = 0.0
    proposer_reward: Optional[float] = None
    solver_reward: Optional[float] = None

    @property
    def reward(self) -> float:
        """
        Unified episode reward:
        - Prefer explicit proposer/solver rewards blended with verifier_score if present.
        - Otherwise use verifier_score.
        - Fallback to binary verified (1.0 / 0.0).
        """
        vs = float(self.verifier_score) if self.verifier_score is not None else None
        pr = float(self.proposer_reward) if self.proposer_reward is not None else None
        sr = float(self.solver_reward) if self.solver_reward is not None else None

        if pr is not None or sr is not None or vs is not None:
            # Blend (tune weights as you like)
            w_pr, w_vs, w_sr = 0.2, 0.6, 0.2
            pr_ = pr if pr is not None else 0.0
            vs_ = vs if vs is not None else (1.0 if self.verified else 0.0)
            sr_ = sr if sr is not None else 0.0
            return max(0.0, min(1.0, w_pr * pr_ + w_vs * vs_ + w_sr * sr_))

        return 1.0 if self.verified else 0.0
