# stephanie/scoring/score_result.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScoreResult:
    dimension: str
    score: float
    rationale: str
    weight: float
    source: str
    target_type: str
    prompt_hash: str

    # SICQL / EBT / HRM fields
    energy: Optional[float] = None
    q_value: Optional[float] = None
    state_value: Optional[float] = None
    policy_logits: Optional[List[float]] = None
    uncertainty: Optional[float] = None
    entropy: Optional[float] = None
    advantage: Optional[float] = None

    # Catch-all for unknown attributes
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreResult":
        known_fields = {
            "dimension", "score", "rationale", "weight", "source",
            "target_type", "prompt_hash", "energy", "q_value", "state_value",
            "policy_logits", "uncertainty", "entropy", "advantage"
        }

        known = {k: data[k] for k in known_fields if k in data}
        extra = {k: v for k, v in data.items() if k not in known_fields}

        return cls(**known, extra=extra)

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "dimension": self.dimension,
            "score": self.score,
            "rationale": self.rationale,
            "weight": self.weight,
            "source": self.source,
            "target_type": self.target_type,
            "prompt_hash": self.prompt_hash,
            "energy": self.energy,
            "q_value": self.q_value,
            "state_value": self.state_value,
            "policy_logits": self.policy_logits,
            "uncertainty": self.uncertainty,
            "entropy": self.entropy,
            "advantage": self.advantage
        }
        return {**base, **self.extra}
