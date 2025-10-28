# stephanie/types/events.py
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, NonNegativeInt, field_validator


class TransitionEvent(BaseModel):
    run_id: str
    step_idx: NonNegativeInt
    agent: str
    state: Dict[str, Any] = Field(default_factory=dict)
    action: Dict[str, Any] = Field(default_factory=dict)
    reward_air: float = 0.0
    rewards_vec: Dict[str, float] = Field(default_factory=dict)

    @field_validator("rewards_vec")
    @classmethod
    def clamp_rew(cls, v):
        # ensure floats and clamp to reasonable range
        out = {}
        for k, val in v.items():
            try:
                x = float(val)
                if x != x:
                    x = 0.0
            except Exception:
                x = 0.0
            out[k] = max(-1e9, min(1e9, x))
        return out


class FragmentRecord(BaseModel):
    case_id: int
    source_type: str
    text: str
    section: Optional[str] = None
    attrs: Dict[str, Any] = Field(default_factory=dict)
    scores: Dict[str, float] = Field(default_factory=dict)
    uncertainty: Optional[float] = None
