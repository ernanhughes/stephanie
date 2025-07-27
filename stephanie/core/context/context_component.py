# stephanie/context/context_component.py
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from stephanie.scoring.score_bundle import ScoreBundle


@dataclass
class ContextComponent:
    name: str
    content: Any
    source: str
    score: Optional[ScoreBundle] = None
    priority: float = 1.0
    timestamp: str = ""
    
    def __post_init__(self):
        """Ensure timestamp is set"""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()