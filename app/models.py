from dataclasses import dataclass
from typing import Optional, List

@dataclass
class MemoryEntry:
    id: int
    title: str
    timestamp: str
    user_text: str
    ai_text: str
    content: str
    summary: str
    tags: List[str]
    source: Optional[str] = None
    openai_url: Optional[str] = None
    importance: Optional[int] = 0
    archived: Optional[bool] = False
    embedding: Optional[List[float]] = None
    tsv: Optional[str] = None
    length: Optional[int] = None
    vector_score: Optional[float] = None
    text_score: Optional[float] = None
    # created_at: str = None
    # updated_at: str = None