from dataclasses import dataclass
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