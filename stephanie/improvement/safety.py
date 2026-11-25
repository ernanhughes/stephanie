"""Fail-closed boundary for model-visible evidence and experience context."""
from __future__ import annotations

import re
from typing import Any


class ProtectedContextError(ValueError):
    """Raised when evaluator-only or untrusted material could reach inference."""


_PROTECTED_KEYS = {
    "oracle", "oracleverified", "oraclelabel", "oracleanswer", "oraclefiles",
    "expectedanswer", "expectedlabel", "expectedoutcome", "expectedfinding",
    "groundtruth", "groundtruthlabel", "referenceanswer", "referencesolution",
    "sealedsolution", "protectedrubric", "answerkey", "hiddenanswer", "hiddenlabel",
    "evaluatoranswer", "evaluationanswer", "correctcandidate", "goldanswer",
}
_BAD_VISIBILITY = {"protected", "evaluator_only", "development_only", "hidden"}
_BAD_INTEGRITY = {"contaminated", "untrusted", "invalid"}


def _normal_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def assert_model_context_safe(value: Any, *, current_experiment_id: str | None = None, path: str = "$",) -> None:
    """Reject protected labels, current evaluation and untrusted payload metadata.

    Callers must use explicit field-whitelisted builders as well; this scan is
    a defense in depth over structured payloads, not semantic prose detection.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normal_key(str(key))
            if normalized in _PROTECTED_KEYS or any(
                normalized.startswith(prefix) for prefix in ("expected", "groundtruth", "sealedanswer", "hiddenanswer", "protectedrubric")
            ) or ("hidden" in normalized and "expected" in normalized) or "oracleverified" in normalized or "oracleanswer" in normalized:
                raise ProtectedContextError(f"protected evaluation field at {path}.{key}")
            if normalized in {"visibility", "dataclassification", "trustboundary"} and str(child).lower() in _BAD_VISIBILITY:
                raise ProtectedContextError(f"non-model-visible content at {path}.{key}")
            if normalized in {"integrity", "integritystatus", "truststatus"} and str(child).lower() in _BAD_INTEGRITY:
                raise ProtectedContextError(f"untrusted content at {path}.{key}")
        if current_experiment_id is not None:
            experiment = value.get("experiment_id") or value.get("current_experiment_id")
            stage = str(value.get("experiment_state") or value.get("record_state") or "").lower()
            if str(experiment) == current_experiment_id and stage in {"current", "running", "evaluation", "evaluated"}:
                raise ProtectedContextError(f"current experiment outcome at {path} is not available to generation")
        for key, child in value.items():
            assert_model_context_safe(child, current_experiment_id=current_experiment_id, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_model_context_safe(child, current_experiment_id=current_experiment_id, path=f"{path}[{index}]")


def validate_reviewed_experience(item: dict) -> None:
    """Require an owner-reviewed historical observation safe for model context."""
    if item.get("reuse_status") != "OWNER_REVIEWED_HISTORICAL":
        raise ProtectedContextError("experience is not approved as historical context")
    if item.get("visibility") != "MODEL_VISIBLE_HISTORY":
        raise ProtectedContextError("experience is not approved for model context")
    if item.get("integrity") != "TRUSTED_SANITIZED":
        raise ProtectedContextError("experience integrity is not qualified")
    assert_model_context_safe(item)
