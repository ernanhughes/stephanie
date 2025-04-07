from dataclasses import dataclass
from typing import Optional, List

@dataclass
class MemoryEntry:
    id: int
    title: str
    timestamp: str
    user_text: str
    ai_text: str
    summary: str
    tags: List[str]
    source: Optional[str] = None
    openai_url: Optional[str] = None
    importance: Optional[int] = 0
    archived: Optional[bool] = False
    # created_at: str = None
    # updated_at: str = None