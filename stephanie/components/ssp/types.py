from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EpisodeStatus(str, Enum):
    PROPOSED = "proposed"
    SOLVED = "solved"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass(frozen=True)
class Proposal:
    query: str
    verification_approach: str
    difficulty: float
    connections: List[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass(frozen=True)
class Solution:
    answer: str
    reasoning_path: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    search_depth: int = 0
    report: Dict[str, Any] = field(default_factory=dict)
    training_batch: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Verification:
    is_valid: bool
    score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    evidence_count: int = 0
    reasoning_steps: int = 0


@dataclass
class Episode:
    id: str
    proposal: Proposal
    solution: Optional[Solution]
    verification: Optional[Verification]
    status: EpisodeStatus
    metrics: Dict[str, float] = field(default_factory=dict)
