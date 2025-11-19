# stephanie/components/vibe/writing_types.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class WritingScore:
    """
    Canonical writing quality score for a single text artifact.

    All fields should be normalized to [0, 1] or [0, 100] consistently.
    v1 assumes [0, 100] to match rubric scores.
    """

    clarity: float
    structure: float
    technical_correctness: float
    depth: float
    actionability: float
    vibe: float

    # Optional aggregate and extra dimensions
    overall: float
    breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure breakdown is present even if None
        d.setdefault("breakdown", {})
        return d
