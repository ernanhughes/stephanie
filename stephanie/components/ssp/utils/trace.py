# stephanie/components/ssp/utils/episode_trace.py
"""
Episode Trace Data Structure

Captures the complete state of an SSP episode, including fields needed
for self-play rewards and VPM visualization.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

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
    reward: float                     # may be 0–1 or 1–100; normalized internally
    solver_steps: int
    difficulty: float = 0.0                   # expected 0–1
    proposer_meta: Dict[str, Any] = field(default_factory=dict)
    verifier_meta: Dict[str, Any] = field(default_factory=dict)
    solver_meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    episode_duration: float = 0.0
    proposer_reward: Optional[float] = None   # expected 0–1
    solver_reward: Optional[float] = None     # expected 
    meta: Dict[str, Any] = field(default_factory=dict)  # extra data

    # ---------------------- helpers & properties ----------------------

    def _norm_vscore(self) -> float:
        """
        Normalize reward to [0,1].
        Accepts 0–1 or 1–100; anything else is clamped.
        """
        vs = 0.0 if self.reward is None else float(self.reward)
        if vs <= 1.0:
            out = vs
        elif vs <= 100.0:
            out = vs / 100.0
        else:
            # out-of-range safeguard: map via sigmoid-ish clamp
            out = 1.0 / (1.0 + math.exp(-(vs - 50.0) / 10.0))
        return max(0.0, min(1.0, out))

    proposer_reward: Optional[float] = None
    solver_reward: Optional[float] = None

    # ---------------------- (de)serialization ----------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed_answer": self.seed_answer,
            "question": self.question,
            "proposer_evidence": list(self.proposer_evidence or []),
            "predicted_answer": self.predicted_answer,
            "evidence_docs": list(self.evidence_docs or []),
            "verified": bool(self.verified),
            "solver_steps": int(self.solver_steps or 0),
            "difficulty": float(self.difficulty or 0.0),
            "timestamp": self.timestamp.isoformat(),
            "episode_duration": float(self.episode_duration or 0.0),
            "proposer_meta": dict(self.proposer_meta or {}),
            "verifier_meta": dict(self.verifier_meta or {}),
            "solver_meta": dict(self.solver_meta or {}),
            "proposer_reward": (None if self.proposer_reward is None else float(self.proposer_reward)),
            "solver_reward": (None if self.solver_reward is None else float(self.solver_reward)),
            "reward": float(self.reward),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpisodeTrace:
        """
        Convenience constructor for reloading. Tolerates ISO timestamp strings.
        """
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                data = dict(data)
                data["timestamp"] = datetime.fromisoformat(ts)
            except Exception:
                data["timestamp"] = datetime.now()
        return cls(
            episode_id=data.get("episode_id", ""),
            seed_answer=data.get("seed_answer", ""),
            question=data.get("question", ""),
            proposer_evidence=list(data.get("proposer_evidence", [])),
            predicted_answer=data.get("predicted_answer", ""),
            evidence_docs=list(data.get("evidence_docs", [])),
            verified=bool(data.get("verified", False)),
            reward=float(data.get("reward", 0.0)),
            solver_steps=int(data.get("solver_steps", 0)),
            difficulty=float(data.get("difficulty", 0.0)),
            proposer_meta=dict(data.get("proposer_meta", {})),
            verifier_meta=dict(data.get("verifier_meta", {})),
            solver_meta=dict(data.get("solver_meta", {})),
            timestamp=data.get("timestamp", datetime.now()),
            episode_duration=float(data.get("episode_duration", 0.0)),
            proposer_reward=data.get("proposer_reward"),
            solver_reward=data.get("solver_reward"),
        )
