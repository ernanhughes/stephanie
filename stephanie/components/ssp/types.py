# stephanie/components/ssp/types.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class Proposal:
    # ❗ Non-defaults first
    query: str
    # defaults after
    difficulty: float = 0.30
    novelty: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_any(cls, x: Any) -> "Proposal":
        """
        Accepts Proposal | dict | (query, difficulty) | [query, difficulty].
        """
        if isinstance(x, Proposal):
            return x
        if isinstance(x, dict):
            # tolerate alternate keys
            q = x.get("query") or x.get("question") or x.get("prompt") or ""
            return cls(
                query=q,
                difficulty=float(x.get("difficulty", 0.30)),
                novelty=float(x.get("novelty", 0.0)),
                meta=dict(x.get("meta", {})),
            )
        if isinstance(x, (list, tuple)) and x:
            q = str(x[0])
            d = float(x[1]) if len(x) > 1 else 0.30
            return cls(query=q, difficulty=d)
        # last resort
        return cls(query=str(x), difficulty=0.30)


@dataclass(slots=True)
class Solution:
    # all defaults → order is fine
    answer: str = ""
    evidence: List[str] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    training_batch: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_any(cls, x: Any) -> "Solution":
        if isinstance(x, Solution):
            return x
        if isinstance(x, dict):
            return cls(
                answer=x.get("answer", "") or x.get("text", "") or "",
                evidence=list(x.get("evidence", []) or []),
                reasoning_path=list(x.get("reasoning_path", []) or []),
                training_batch=x.get("training_batch"),
                meta=dict(x.get("meta", {})),
            )
        return cls(answer=str(x))


@dataclass(slots=True)
class Verification:
    # ❗ Non-defaults first
    is_valid: bool
    score: float
    # defaults after
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    evidence_count: int = 0
    reasoning_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
