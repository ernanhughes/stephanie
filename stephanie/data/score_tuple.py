
# stephanie/data/score_tuple.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ScoreTuple:
    score: float
    count: int
    dimension: str
    sub_dimension: str
    meta: Dict[str, Any] = field(default_factory=dict)
 