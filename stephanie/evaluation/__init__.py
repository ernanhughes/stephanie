# stephanie/evaluation/__init__.py
"""Canonical Evaluation Runtime (Stage 2).

measurement != interpretation != decision.
"""
from __future__ import annotations

from stephanie.evaluation.compare import (
    CANONICAL_ONLY,
    EXPECTED_DIVERGENCE,
    LEGACY_ONLY,
    MATCH,
    UNEXPECTED_DIVERGENCE,
    ComparisonOutcome,
    compare_evaluations,
)
from stephanie.evaluation.context import EvaluationContext
from stephanie.evaluation.confidence import (
    BOOTSTRAP,
    CALIBRATION_MODEL,
    ENSEMBLE_AGREEMENT,
    EVIDENCE_POLICY,
    HUMAN,
    JUDGE_SELF_REPORT,
    RULE,
    is_measured,
    validate_confidence,
)
from stephanie.evaluation.criterion import Criterion, ScoreScale
from stephanie.evaluation.diagnostics import (
    CONFIDENCE_AMBIGUOUS,
    DUAL_READ_DIVERGENCE,
    DUAL_WRITE_FAILURE,
    FUSION_VERSION_MISSING,
    IDENTITY_UNRESOLVED,
    LEGACY_SNAPSHOT_DIVERGENCE,
    LEGACY_WRITE_BROKEN,
    ORPHAN_ATTRIBUTE,
    SCALE_UNKNOWN,
    VERDICT_AMBIGUOUS,
    EvaluationDiagnostic,
)
from stephanie.evaluation.evaluation import (
    Evaluation,
    EvaluationAttribute,
    EvaluationObservation,
    EvaluatorRef,
)
from stephanie.evaluation.evidence import EvaluationEvidenceLink, EvidenceRef
from stephanie.evaluation.fusion import FusedScore, FusionSpec, fuse_weighted_mean
from stephanie.evaluation.interpretation import Interpretation
from stephanie.evaluation.provenance import EvaluationProvenance
from stephanie.evaluation.repository import (
    EvaluationReader,
    EvaluationWriter,
    InMemoryEvaluationRepository,
)
from stephanie.evaluation.score import Score, ScoreAttribute
from stephanie.evaluation.subject import SubjectRef

__all__ = [
    "SubjectRef",
    "Criterion",
    "ScoreScale",
    "EvaluatorRef",
    "Evaluation",
    "EvaluationObservation",
    "EvaluationAttribute",
    "Score",
    "ScoreAttribute",
    "EvidenceRef",
    "EvaluationEvidenceLink",
    "EvaluationProvenance",
    "Interpretation",
    "FusionSpec",
    "FusedScore",
    "fuse_weighted_mean",
    "EvaluationReader",
    "EvaluationWriter",
    "InMemoryEvaluationRepository",
    "ComparisonOutcome",
    "compare_evaluations",
    "EvaluationContext",
    "EvaluationDiagnostic",
    "validate_confidence",
    "is_measured",
    "MATCH",
    "EXPECTED_DIVERGENCE",
    "UNEXPECTED_DIVERGENCE",
    "LEGACY_ONLY",
    "CANONICAL_ONLY",
    "LEGACY_SNAPSHOT_DIVERGENCE",
    "SCALE_UNKNOWN",
    "CONFIDENCE_AMBIGUOUS",
    "VERDICT_AMBIGUOUS",
    "ORPHAN_ATTRIBUTE",
    "IDENTITY_UNRESOLVED",
    "LEGACY_WRITE_BROKEN",
    "FUSION_VERSION_MISSING",
    "DUAL_READ_DIVERGENCE",
    "DUAL_WRITE_FAILURE",
    "JUDGE_SELF_REPORT",
    "ENSEMBLE_AGREEMENT",
    "BOOTSTRAP",
    "CALIBRATION_MODEL",
    "EVIDENCE_POLICY",
    "RULE",
    "HUMAN",
]
