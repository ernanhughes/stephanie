# stephanie/evaluation/subject.py
"""Canonical subject identity (§4). Strings only; no domain FK graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SubjectRef:
    subject_type: str
    subject_id: str

    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce int PKs etc. — canonical keys are always strings.
        object.__setattr__(self, "subject_type", str(self.subject_type))
        object.__setattr__(self, "subject_id", str(self.subject_id))

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject_type, self.subject_id)
