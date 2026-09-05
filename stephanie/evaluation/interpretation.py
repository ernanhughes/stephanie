# stephanie/evaluation/interpretation.py
"""Evaluation-level interpretation (§12). Namespaced; never an experiment decision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Interpretation:
    namespace: str
    value: str
    rationale: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.namespace}:{self.value}"
