from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Hypothesis:
    id: Optional[int] = None
    goal: str = ""                            # Required
    goal_type: str = ""
    text: str = ""                            # Required
    prompt: str = ""
    confidence: float = 0.0
    pipeline_signature: Optional[str] = None
    review: Optional[str] = None
    reflection: Optional[str] = None
    elo_rating: float = 750.0
    embedding: Optional[list] = field(default_factory=list)  # Expecting a 1024-dim vector
    features: Optional[dict[str, any]] = field(default_factory=dict)
    source_hypothesis: Optional[int] = None
    strategy_used: Optional[str] = None
    version: int = 1
    source: Optional[str] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)