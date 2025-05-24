from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Goal:
    goal_text: str
    id: Optional[int] = None
    goal_type: Optional[str] = None
    focus_area: Optional[str] = None
    strategy: Optional[str] = None
    llm_suggested_strategy: Optional[str] = None
    source: str = "user"
    created_at: Optional[datetime] = None