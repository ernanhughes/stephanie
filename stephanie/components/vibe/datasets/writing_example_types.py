from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class WritingDatasetRow:
    text: str
    clarity: float
    structure: float
    technical_correctness: float
    depth: float
    actionability: float
    vibe: float
    overall: float

    # provenance / metadata
    conversation_id: str
    turn_index: int
    role: str
    tags: str  # comma-separated or raw JSON string
    extra_meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # serialize extra_meta as string if you want a flat CSV
        return d
