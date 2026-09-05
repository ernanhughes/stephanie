# stephanie/evaluation/fusion.py
"""Derived fusion over raw scores (§16). Raw scores stay authoritative."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from stephanie.evaluation.score import Score


@dataclass(frozen=True)
class FusionSpec:
    fusion_id: str
    version: str

    method: str  # e.g. "weighted_mean"

    weights: Mapping[str, float] = field(default_factory=dict)

    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedScore:
    value: float
    confidence: Optional[float]

    fusion_spec_id: str

    component_score_ids: tuple[str, ...]


def fuse_weighted_mean(scores: Sequence[Score], spec: FusionSpec) -> FusedScore:
    """Recomputable weighted mean. Skips None weights as 1.0; no silent rescaling."""
    total = 0.0
    weight_sum = 0.0
    ids: list[str] = []
    for score in scores:
        weight = spec.weights.get(score.dimension, score.weight if score.weight is not None else 1.0)
        total += score.value * weight
        weight_sum += weight
        ids.append(score.score_id)
    if not ids:
        raise ValueError("cannot fuse an empty score set")
    value = total / weight_sum if weight_sum else 0.0
    measured = [s.confidence for s in scores if s.confidence is not None]
    confidence = sum(measured) / len(measured) if measured else None
    return FusedScore(
        value=value,
        confidence=confidence,
        fusion_spec_id=f"{spec.fusion_id}@{spec.version}",
        component_score_ids=tuple(ids),
    )
