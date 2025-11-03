# stephanie/types/lightning.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, NonNegativeInt

LightningKind = Literal[
    "heartbeat",         # periodic ping (progress)
    "candidate",         # a fresh child/candidate was created
    "score_update",      # a node got evaluated
    "leaderboard",       # current top-K leaves
    "checkpoint",        # N-th simulation checkpoint
    "final"              # final winners
]

class LightningResult(BaseModel):
    run_id: str
    step_idx: NonNegativeInt
    kind: LightningKind
    agent: str
    payload: Dict[str, Any] = Field(default_factory=dict)
