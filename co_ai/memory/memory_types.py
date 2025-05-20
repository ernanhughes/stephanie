from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Goal:
    id: Optional[int]
    goal_text: str
    goal_type: Optional[str] = None
    focus_area: Optional[str] = None
    strategy: Optional[str] = None
    llm_suggested_strategy: Optional[str] = None
    source: str = "user"
    created_at: Optional[datetime] = None

@dataclass
class MRQMemoryEntry:
    goal: str
    agent_name: str
    strategy: str
    prompt: str
    response: str
    reward: float
    prompt_embedding: Optional[list[float]] = None
    response_embedding: Optional[list[float]] = None
    review_embedding: Optional[list[float]] = None
    reflection_embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    id: Optional[int] = None  # Assigned by the DB

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