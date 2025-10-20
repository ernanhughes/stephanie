# stephanie/components/gap/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GapConfig:
    """Configuration for GAP analysis."""
    dimensions: List[str] = field(default_factory=lambda: [
        "reasoning", "knowledge", "clarity", "faithfulness", "coverage"
    ])
    hrm_scorers: List[str] = field(default_factory=lambda: ["hf_hrm"])
    tiny_scorers: List[str] = field(default_factory=lambda: ["hf_mistral"])
    # hrm_scorers: List[str] = field(default_factory=lambda: ["hrm"])
    # tiny_scorers: List[str] = field(default_factory=lambda: ["tiny"])
    out_dir: Path = field(default_factory=lambda: Path("data/gap_runs/vpm"))
    base_dir: Path = field(default_factory=lambda: Path("data/gap_runs"))
    interleave: bool = False
    progress_log_every: int = 25
    dedupe_policy: str = "first_wins"
    per_dim_cap: int = 1000 # CCAP count limit per dimension
    # per_dim_cap: int = 100 
    route_threshold_uncertainty: float = 0.6
    route_threshold_ood: float = 0.7
    enable_scm_head: bool = True
    scm: Dict[str, Any] = field(default_factory=lambda: {
        "reasoning_dimensions": ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"],
        "latent_dim": 128,
    })

@dataclass
class TripleSample:
    """A single training sample for comparison."""
    node_id: str
    dimension: str
    goal_text: str
    output_text: str
    target_value: Optional[float] = None
    fingerprint: Optional[str] = None

@dataclass
class ModelScores:
    """Scores for a single model across dimensions."""
    model_name: str
    scores: Dict[str, float]
    metrics: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None