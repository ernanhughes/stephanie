# stephanie/evaluation/score.py
"""Canonical Score (§9) + score attributes (§18). Raw measurements only."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional

from stephanie.evaluation.confidence import validate_confidence
from stephanie.evaluation.criterion import ScoreScale


@dataclass(frozen=True)
class Score:
    score_id: str
    evaluation_id: str

    dimension: str
    value: float

    scale: Optional[ScoreScale] = None

    weight: Optional[float] = None

    confidence: Optional[float] = None
    confidence_source: Optional[str] = None

    scorer: Optional[str] = None
    source: Optional[str] = None

    rationale: Optional[str] = None

    created_at: Optional[datetime] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_confidence(self.confidence)
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class ScoreAttribute:
    score_id: str

    namespace: str
    name: str

    value: Any

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}.{self.name}"
