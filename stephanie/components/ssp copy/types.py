# stephanie/components/ssp/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Proposal:
    query: str
    verification_approach: str
    difficulty: float
    connections: List[str] = field(default_factory=list)
    prior: float = 1.0
    raw_response: str = ""
    path_id: str = ""
    depth: int = 0

@dataclass
class Solution:
    answer: str
    reasoning_path: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    search_depth: int = 0
    trace_id: str = ""

@dataclass
class Verification:
    is_valid: bool
    score: float
    dimension_scores: Dict[str, float]
    evidence_count: int
    reasoning_steps: int
    checks: Dict[str, bool] = field(default_factory=dict)

@dataclass
class RewardBreakdown:
    hrm_delta: float
    mars_delta: float
    verifier_bonus: float
    length_penalty: float

@dataclass
class SensoryBundle:
    vpm: Dict[str, Any]               # {tensor, significance_map, metadata}
    scm: Dict[str, float]             # {coherence, novelty, complexity, ...}
    epistemic: Dict[str, float]       # {difficulty, success_rate, growth, verification_rate}
    meta: Dict[str, Any] = field(default_factory=dict)
