from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class SharpeningPrediction:
    id: Optional[int]
    goal_id: int
    prompt_text: str
    output_a: str
    output_b: str
    preferred: str  # 'a' or 'b'
    predicted: str  # 'a' or 'b'
    value_a: float
    value_b: float
    created_at: Optional[datetime] = None