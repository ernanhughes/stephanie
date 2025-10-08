# stephanie/utils/metrics_schema.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class MetricVector:
    """Immutable container for a metrics vector + names in aligned order."""
    names: Tuple[str, ...]
    values: Tuple[float, ...]

    def to_dict(self) -> Dict[str, float]:
        return dict(zip(self.names, self.values))

    def extend(self, other: "MetricVector") -> "MetricVector":
        return MetricVector(
            names=self.names + other.names,
            values=self.values + other.values,
        )

def flatten_score_bundle(bundle: Dict[str, Any]) -> MetricVector:
    """
    Flattens a typical scoring 'bundle' you already use (per AgentScorerAgent)
    into deterministic (names, values) tuples. 
    - Supports nested dicts of primitives (floats/ints/bools)
    - Dot-joins keys for stability, e.g. 'ebt.relevance.score'
    - Booleans → 0/1; non-numeric primitives are skipped
    """
    names: List[str] = []
    values: List[float] = []

    def walk(prefix: str, obj: Any):
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                walk(f"{prefix}.{k}" if prefix else k, obj[k])
        elif isinstance(obj, (int, float)):
            names.append(prefix)
            values.append(float(obj))
        elif isinstance(obj, bool):
            names.append(prefix)
            values.append(1.0 if obj else 0.0)
        # strings or complex types are intentionally skipped here

    walk("", bundle or {})
    return MetricVector(tuple(names), tuple(values))

def concat_vectors(parts: List[MetricVector]) -> MetricVector:
    names: List[str] = []
    values: List[float] = []
    for p in parts:
        names.extend(p.names)
        values.extend(p.values)
    return MetricVector(tuple(names), tuple(values))