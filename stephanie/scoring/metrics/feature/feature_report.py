# stephanie/scoring/metrics/feature/feature_report.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FeatureReport:
    name: str
    kind: str                  # "row" | "group"
    ok: bool
    quality: float | None = None
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
