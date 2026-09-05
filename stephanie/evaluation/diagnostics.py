# stephanie/evaluation/diagnostics.py
"""Explicit migration diagnostics (§35). Never repair ambiguous data silently."""
from __future__ import annotations

LEGACY_SNAPSHOT_DIVERGENCE = "EVAL_LEGACY_SNAPSHOT_DIVERGENCE"
SCALE_UNKNOWN = "EVAL_SCALE_UNKNOWN"
CONFIDENCE_AMBIGUOUS = "EVAL_CONFIDENCE_AMBIGUOUS"
VERDICT_AMBIGUOUS = "EVAL_VERDICT_AMBIGUOUS"
ORPHAN_ATTRIBUTE = "EVAL_ORPHAN_ATTRIBUTE"
IDENTITY_UNRESOLVED = "EVAL_IDENTITY_UNRESOLVED"
LEGACY_WRITE_BROKEN = "EVAL_LEGACY_WRITE_BROKEN"
FUSION_VERSION_MISSING = "EVAL_FUSION_VERSION_MISSING"
DUAL_READ_DIVERGENCE = "EVAL_DUAL_READ_DIVERGENCE"
DUAL_WRITE_FAILURE = "EVAL_DUAL_WRITE_FAILURE"


class EvaluationDiagnostic(Exception):
    """Carries a diagnostic code plus context (collected, not raised, by harnesses)."""

    def __init__(self, code: str, message: str, context: dict | None = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.context = dict(context or {})
