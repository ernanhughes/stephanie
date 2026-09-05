# stephanie/evaluation/confidence.py
"""One canonical confidence semantic (§10–§11).

Confidence = uncertainty about the reliability of an observation,
0.0–1.0, source-labeled, nullable. None means unmeasured — never zero.
"""
from __future__ import annotations

from typing import Optional

# Known confidence origins (open set; callers may add their own labels).
JUDGE_SELF_REPORT = "judge_self_report"
ENSEMBLE_AGREEMENT = "ensemble_agreement"
BOOTSTRAP = "bootstrap"
CALIBRATION_MODEL = "calibration_model"
EVIDENCE_POLICY = "evidence_policy"
RULE = "rule"
HUMAN = "human"


def validate_confidence(value: Optional[float]) -> Optional[float]:
    """Return value if valid; raise on out-of-range; pass None through."""
    if value is None:
        return None
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {value!r}")
    return float(value)


def is_measured(value: Optional[float]) -> bool:
    return value is not None
