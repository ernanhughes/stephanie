from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class EpisodeTrace:
    episode_id: str
    seed_answer: str
    question: str
    predicted_answer: str
    verified: bool
    reward: float
    difficulty: float = 0.0
    solver_steps: int = 0
    evidence_docs: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

