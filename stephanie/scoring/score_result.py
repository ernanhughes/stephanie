# stephanie/scoring/score_result.py
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ScoreResult:
    dimension: str
    score: float
    rationale: str
    weight: float
    source: str
    target_type: str
    prompt_hash: str
    # SICQL-specific fields
    energy: Optional[float] = None
    q_value: Optional[float] = None
    state_value: Optional[float] = None
    policy_logits: Optional[list[float]] = None
    uncertainty: Optional[float] = None
    entropy: Optional[float] = None
    advantage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "rationale": self.rationale,
            "weight": self.weight,
            "energy": self.energy,
            "source": self.source,
            "target_type": self.target_type,
            "prompt_hash": self.prompt_hash,
            "q_value": self.q_value,
            "state_value": self.state_value,
            "policy_logits": self.policy_logits,
            "uncertainty": self.uncertainty,
            "entropy": self.entropy,
            "advantage": self.advantage
        }