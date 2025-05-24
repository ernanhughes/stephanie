from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from datetime import datetime

@dataclass
class ReflectionDelta:
    id: Optional[int] = None
    goal_id: int = 0
    run_id_a: str = ""
    run_id_b: str = ""
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    score_delta: Optional[float] = None
    pipeline_a: Optional[Dict] = None
    pipeline_b: Optional[Dict] = None
    pipeline_diff: Optional[Dict] = None  # {"only_in_a": [...], "only_in_b": [...]}
    strategy_diff: Optional[bool] = False
    model_diff: Optional[bool] = False
    rationale_diff: Optional[Tuple[str, str]] = None
    created_at: Optional[datetime] = None
