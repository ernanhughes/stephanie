from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class RewardItem:
    prompt: str
    response: str
    ground_truth: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None  # evidence, tools_used, etc.

@dataclass
class RewardOutput:
    reward: float
    features: Dict[str, float]           # e.g., {"f1":0.63, "coverage":0.41, ...}
    version: str = "1"
    schema: str = "qa_v1"

class RewardHead(Protocol):
    async def score(self, item: RewardItem) -> RewardOutput: ...
    async def score_batch(self, items: List[RewardItem]) -> List[RewardOutput]: ...