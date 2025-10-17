# stephanie/components/gap/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import time

@dataclass
class GapRunManifest:
    """Represents a GAP analysis run with all metadata."""
    run_id: str
    dataset: str
    models: Dict[str, str]
    dimensions: List[str]
    preproc_version: str = "v1"
    created_at: float = field(default_factory=time.time)
    
    # Paths will be populated during run
    paths: Dict[str, str] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GapConfig:
    """Configuration for GAP analysis."""
    dimensions: List[str] = field(default_factory=lambda: [
        "reasoning", "knowledge", "clarity", "faithfulness", "coverage"
    ])
    hrm_scorers: List[str] = field(default_factory=lambda: ["hrm"])
    tiny_scorers: List[str] = field(default_factory=lambda: ["tiny"])
    out_dir: Path = field(default_factory=lambda: Path("data/vpm"))
    base_dir: Path = field(default_factory=lambda: Path("data/gap_runs"))
    interleave: bool = False
    progress_log_every: int = 25
    dedupe_policy: str = "first_wins"
    per_dim_cap: int = 100
    route_threshold_uncertainty: float = 0.6
    route_threshold_ood: float = 0.7

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