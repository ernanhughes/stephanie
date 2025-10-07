# stephanie/metrics/providers/base.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Protocol, List
from dataclasses import dataclass

@dataclass(frozen=True)
class MetricVector:
    names: Tuple[str, ...]
    values: Tuple[float, ...]

class MetricProvider(Protocol):
    name: str      # short, stable; e.g. "basic", "embed", "scorer", "mars"
    version: str   # semver string, e.g. "1.0.0"

    async def compute(self, *, goal: str, text: str, context: Dict[str, Any]) -> MetricVector:
        ...
