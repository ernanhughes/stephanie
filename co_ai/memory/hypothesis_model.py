# co_ai/memory/hypothesis_model.py
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class Hypothesis:
    goal: str
    text: str
    confidence: Optional[float] = None
    review: Optional[str] = None
    embedding: Optional[List[float]] = None
    features: Optional[List[str]] = None
    created_at: str = datetime.now(timezone.utc).isoformat()

    def to_dict(self):
        return asdict(self)

    def short_summary(self):
        return f"Hypothesis: {self.text} (Confidence: {self.confidence or 'N/A'})"
