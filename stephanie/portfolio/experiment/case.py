# stephanie/portfolio/experiment/case.py
"""Frozen benchmark case (§Experiment 001: freeze the inputs)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ExpectedFinding:
    code: str
    keywords: tuple[str, ...]
    severity: str = "major"  # critical | major | minor
    note: str = ""


@dataclass(frozen=True)
class PortfolioBenchmarkCase:
    case_id: str
    task_type: str
    prompt: str
    source_text: str

    expected: tuple[ExpectedFinding, ...] = ()

    source_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def freeze(case_id: str, task_type: str, prompt: str, source_text: str,
               expected: tuple[ExpectedFinding, ...] = (),
               metadata: Optional[dict] = None) -> "PortfolioBenchmarkCase":
        digest = hashlib.sha256(f"{task_type}\n{prompt}\n{source_text}".encode()).hexdigest()[:16]
        return PortfolioBenchmarkCase(
            case_id=case_id, task_type=task_type, prompt=prompt,
            source_text=source_text, expected=expected,
            source_hash=f"sha256:{digest}", metadata=dict(metadata or {}),
        )
