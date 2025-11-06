# stephanie/arena/plugins/interfaces.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class JobCtx:
    job_id: str
    task_type: str            # e.g., "expository-blog", "howto", ...
    topic: str                # e.g., "hierarchical reasoning"
    season: Optional[str] = None
    extras: Dict[str, Any] = None  # freeform routing features

@dataclass
class PlayResult:
    artefact_id: Optional[int]     # e.g., BlogDraft.id
    artefact_type: str             # e.g., "blog_draft"
    metrics: Dict[str, float]      # raw metrics to be scored
    payload: Dict[str, Any]        # freeform (e.g., draft_md)
    ok: bool

class Play(Protocol):
    """A pluggable pipeline that produces an artefact + metrics."""
    name: str
    def run(self, ctx: JobCtx) -> PlayResult: ...

class Scorer(Protocol):
    """Turns PlayResult.metrics into scalar(s)."""
    name: str
    def score(self, ctx: JobCtx, metrics: Dict[str, float]) -> Dict[str, float]: ...
    # return a dict of named scores; "primary" will be used for MCTS reward

class RewardAggregator(Protocol):
    """Aggregate multi-score dict to a single reward scalar for MCTS."""
    def aggregate(self, scores: Dict[str, float]) -> float: ...
