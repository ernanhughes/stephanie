from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SymbolicNode:
    step_name: str
    action: str
    thought: Optional[str] = None
    prompt: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    score: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, str]] = field(default_factory=dict)
