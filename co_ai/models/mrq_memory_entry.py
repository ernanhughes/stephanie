from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


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