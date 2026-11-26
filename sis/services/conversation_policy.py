"""Deterministic per-turn data eligibility for conversational inference."""

from __future__ import annotations

import re
from dataclasses import dataclass


_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|oc_sk_[A-Za-z0-9_-]{12,})\b"),
    re.compile(r"(?i)\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    re.compile(r"(?i)\b(?:password|passwd|token|api[_-]?key|access[_-]?token|client[_-]?secret|private[_-]?key)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(?:BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY)\b"),
)
_PROTECTED_PATTERNS = (
    re.compile(r"(?i)\bprotected evaluation\b"),
    re.compile(r"(?i)\b(?:gold|answer) key\b.{0,60}\b(?:evaluation|benchmark|exam)\b"),
    re.compile(r"(?i)\b(?:evaluation|benchmark|exam)\b.{0,60}\b(?:gold|answer) key\b"),
)

DATA_CLASSES = {
    "approved_external",
    "private_project",
    "protected_evaluation",
    "credential",
    "unknown",
}


@dataclass(frozen=True)
class DataEligibility:
    data_class: str
    external_allowed: bool
    reason: str


class ProtectedContentError(ValueError):
    """Content that must not be sent to any conversational model."""


def contains_credential(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _SECRET_PATTERNS)


def contains_protected_evaluation(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _PROTECTED_PATTERNS)


def classify_turn(
    *,
    current_text: str,
    history,
    external_approved: bool,
    has_project_context: bool = False,
    external_project_approved: bool = False,
    retrieved_context: str = "",
) -> DataEligibility:
    """Classify the assembled model context, conservatively including legacy rows."""
    texts = [current_text]
    classes = []
    for message in history:
        text = str(getattr(message, "text", "") or "")
        if text:
            texts.append(text)
            meta = getattr(message, "meta", None) or {}
            classes.append(meta.get("data_class", "unknown") if isinstance(meta, dict) else "unknown")

    if retrieved_context:
        texts.append(retrieved_context)
    if any(contains_credential(text) for text in texts):
        raise ProtectedContentError("Credential-like content is blocked before inference")
    if any(contains_protected_evaluation(text) for text in texts):
        raise ProtectedContentError("Protected evaluation content is blocked before inference")
    if has_project_context and not external_project_approved:
        return DataEligibility("private_project", False, "The selected project context is not approved for external inference")

    all_history_external = all(value == "approved_external" for value in classes)
    if external_approved and all_history_external and (not has_project_context or external_project_approved):
        return DataEligibility("approved_external", True, "Owner approved this turn and all included project categories for the configured external route")
    if classes and not all_history_external:
        return DataEligibility("unknown", False, "Conversation contains unknown or private earlier turns")
    return DataEligibility("unknown", False, "External use was not explicitly approved for this turn")
