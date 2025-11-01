# stephanie/components/ssp/utils/episode_trace.py
"""
Episode Trace Data Structure

Captures the complete state of an SSP episode, including fields needed
for self-play rewards and VPM visualization.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

import math

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
    verifier_score: float                     # may be 0–1 or 1–100; normalized internally
    solver_steps: int
    difficulty: float = 0.0                   # expected 0–1
    proposer_meta: Dict[str, Any] = field(default_factory=dict)
    verifier_meta: Dict[str, Any] = field(default_factory=dict)
    solver_meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    episode_duration: float = 0.0
    proposer_reward: Optional[float] = None   # expected 0–1
    solver_reward: Optional[float] = None     # expected 0–1

    # ---------------------- helpers & properties ----------------------

    def _norm_vscore(self) -> float:
        """
        Normalize verifier_score to [0,1].
        Accepts 0–1 or 1–100; anything else is clamped.
        """
        vs = 0.0 if self.verifier_score is None else float(self.verifier_score)
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

    @property
    def reward(self) -> float:
        vs = None if self.verifier_score is None else float(self.verifier_score)
        pr = None if self.proposer_reward is None else float(self.proposer_reward)
        sr = None if self.solver_reward is None else float(self.solver_reward)
        # Bring verifier_score onto [0,1] if it looks like a percentage
        if vs is not None and vs > 1.0:
            vs = max(0.0, min(1.0, vs / 100.0))
        if pr is not None and pr > 1.0:
            pr = max(0.0, min(1.0, pr / 100.0))
        if sr is not None and sr > 1.0:
            sr = max(0.0, min(1.0, sr / 100.0))

        if pr is not None or sr is not None or vs is not None:
            w_pr, w_vs, w_sr = 0.2, 0.6, 0.2
            pr_ = 0.0 if pr is None else pr
            vs_ = (1.0 if self.verified else 0.0) if vs is None else vs
            sr_ = 0.0 if sr is None else sr
            return max(0.0, min(1.0, w_pr * pr_ + w_vs * vs_ + w_sr * sr_))
        return 1.0 if self.verified else 0.0

    def to_vpm_features(self) -> Tuple[List[str], List[float]]:
        """Return (names, values) scaled to [0,1] with safe clamps."""
        q_len = float(len((self.question or "").split()))
        a_len = float(len((self.predicted_answer or "").split()))
        ev_count = float(len(self.evidence_docs or []))
        steps = float(self.solver_steps or 0)

        def n01(x, hi):
            return max(0.0, min(1.0, (x / hi) if hi > 0 else 0.0))

        def nlog(x, hi):
            return 0.0 if hi <= 0 else max(0.0, min(1.0, math.log1p(x) / math.log1p(hi)))

        # Normalize verifier score from 0–100 → 0–1 if necessary
        vscore = float(self.verifier_score or 0.0)
        if vscore > 1.0:
            vscore = vscore / 100.0
        vscore = max(0.0, min(1.0, vscore))

        diff = float(self.difficulty or 0.0)
        diff = max(0.0, min(1.0, diff))

        names = [
            "verifier_score",
            "verified",
            "difficulty",
            "question_len",
            "answer_len",
            "evidence_count",
            "solver_steps",
        ]
        vals = [
            vscore,
            1.0 if self.verified else 0.0,
            diff,
            n01(q_len, 128.0),
            n01(a_len, 128.0),
            nlog(ev_count, 64.0),
            nlog(steps, 64.0),
        ]
        _logger.info("VPM features: %s", dict(zip(names, [round(x,4) for x in vals])))
        return names, vals

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
            "verifier_score": float(self.verifier_score or 0.0),
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
            verifier_score=float(data.get("verifier_score", 0.0)),
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
