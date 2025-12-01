# stephanie/components/vibe/vibe.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from stephanie.components.vibe.artifact_snapshot import ArtifactSnapshot
from stephanie.components.vibe.concept_transform import ConceptTransform
from stephanie.components.vibe.test_probe import TestProbe
from stephanie.memcube.rubric import RubricMemCube


@dataclass
class VibeScore:
    functional: float              # 0–1 or 0–100
    vibe: float                    # composite “style / ethics / clarity”
    robustness: float              # CCS / perturbation stability
    cost: float                    # normalized cost (time/compute/$)
    risk: float                    # failure / harm / brittleness

    breakdown: Dict[str, float]    # optional extra dims (e.g. ethics, doc, structure)

@dataclass
class VibeFeatureTask:
    base_state: ArtifactSnapshot
    user_feature: str
    f2p_tests: List[TestProbe]
    p2p_tests: List[TestProbe]
    rubrics: List[RubricMemCube]
    perturbations: List[ConceptTransform]
    context_complexity: float = 0.0
    aggressiveness_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VibeCandidate:
    task: VibeFeatureTask
    candidate_state: ArtifactSnapshot   # result after applying a change
    generator_meta: Dict[str, Any]      # which agent, which policy, cfg hash, etc.

