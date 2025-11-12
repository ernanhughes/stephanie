from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import time
import uuid


class ScoringJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scorable: Dict[str, Any]  # serialized Scorable .to_dict()
    context: Dict[str, Any]  # evaluation context (goal, pipeline_run_id, etc.)
    scorers: List[str] = ["sicql", "mrq", "hrm"]
    dimensions: List[str] = [
        "alignment",
        "faithfulness",
        "coverage",
        "clarity",
        "coherence",
    ]
    scorer_weights: Dict[str, float] = {}
    dimension_weights: Dict[str, float] = {}
    include_llm_heuristic: bool = False
    include_vpm_phi: bool = False
    fuse_mode: str = "weighted_mean"
    clamp_01: bool = True
    return_topic: Optional[str] = None
    created_ts: float = Field(default_factory=lambda: time.time())


def to_kwargs(self) -> Dict[str, Any]:
    return {
        "scorable": self.scorable,
        "context": self.context,
        "scorers": self.scorers,
        "dimensions": self.dimensions,
        "scorer_weights": self.scorer_weights,
        "dimension_weights": self.dimension_weights,
        "include_llm_heuristic": self.include_llm_heuristic,
        "include_vpm_phi": self.include_vpm_phi,
        "fuse_mode": self.fuse_mode,
        "clamp_01": self.clamp_01,
    }
