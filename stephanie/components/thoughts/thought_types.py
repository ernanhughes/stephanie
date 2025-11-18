from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ThoughtKind(str, Enum):
    THINK = "think" # free reasoning / idea
    VERIFY = "verify" # check claim / retrieval
    REPAIR = "repair" # fix inconsistency / gap
    DECIDE = "decide" # choose action / answer


@dataclass
class Evidence:
    source: str # e.g., "web:doi:...", "memcube:...", "trace:step#"
    text: str
    weight: float = 1.0


@dataclass
class Thought:
    text: str
    kind: ThoughtKind = ThoughtKind.THINK
    score: float = 0.0 # calibrated usefulness / quality ([-1, 1] or [0,1])
    uncertainty: float = 1.0 # lower is better
    tags: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    meta: Dict[str, float] = field(default_factory=dict) # energy, advantage, risk, etc.
    created_ts: float = field(default_factory=lambda: time.time())


def to_dict(self) -> Dict:
    return {
        "text": self.text,
        "kind": self.kind.value,
        "score": float(self.score),
        "uncertainty": float(self.uncertainty),
        "tags": list(self.tags),
        "evidence": [e.__dict__ for e in self.evidence],
        "meta": {k: float(v) for k, v in self.meta.items()},
        "created_ts": float(self.created_ts),
    }