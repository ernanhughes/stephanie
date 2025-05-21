from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class SharpeningResult:
    goal: str
    prompt: str
    template: str
    original_output: str
    sharpened_output: str
    preferred_output: str
    winner: str
    improved: bool
    comparison: str
    score_a: float
    score_b: float
    score_diff: float
    best_score: float
    prompt_template: Optional[str] = None
    created_at: Optional[datetime] = None