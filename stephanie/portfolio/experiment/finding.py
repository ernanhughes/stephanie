# stephanie/portfolio/experiment/finding.py
"""Issue-level findings (§Experiment 001: what constitutes a useful finding)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class FindingClass(str, Enum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    DUPLICATE = "DUPLICATE"
    UNVERIFIABLE = "UNVERIFIABLE"
    NOT_ACTIONABLE = "NOT_ACTIONABLE"


@dataclass(frozen=True)
class Finding:
    finding_id: str
    case_id: str
    arm: str
    candidate_id: str

    category: str
    claim: str
    location: Optional[str] = None
    severity: str = "major"

    evidence: tuple[str, ...] = ()
    classification: FindingClass = FindingClass.UNVERIFIABLE
    matched_code: Optional[str] = None
    unique: bool = False

    metadata: Mapping[str, Any] = field(default_factory=dict)
