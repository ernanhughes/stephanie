# stephanie/evaluation/evidence.py
"""Generic evidence layer (§14). Adapted from Writer's evidence/provenance subsystem."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str

    evidence_type: str

    source_id: Optional[str] = None
    source_type: Optional[str] = None

    content_hash: Optional[str] = None

    trust: Optional[float] = None

    captured_at: Optional[datetime] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationEvidenceLink:
    evaluation_id: str
    evidence_id: str
    relationship: str = "supports"  # supports | contradicts | context | source | verification
