from types import SimpleNamespace

import pytest

from sis.services.conversation_policy import (
    ProtectedContentError,
    classify_turn,
)


def test_external_requires_explicit_approval_for_the_complete_history():
    assert classify_turn(
        current_text="Synthetic question",
        history=[],
        external_approved=False,
    ).data_class == "unknown"
    eligible = SimpleNamespace(text="Synthetic earlier turn", meta={"data_class": "approved_external"})
    result = classify_turn(
        current_text="Synthetic follow-up",
        history=[eligible],
        external_approved=True,
    )
    assert result.data_class == "approved_external"
    assert result.external_allowed


def test_unknown_legacy_and_project_context_stay_local():
    legacy = SimpleNamespace(text="Old unclassified conversation", meta={})
    assert classify_turn(
        current_text="A follow-up", history=[legacy], external_approved=True
    ).data_class == "unknown"
    assert classify_turn(
        current_text="Synthetic", history=[], external_approved=True,
        has_project_context=True,
    ).data_class == "private_project"


@pytest.mark.parametrize("text", [
    "api_key=sk-123456789012345678901234",
    "Bearer abcdefghijklmnopqrstuvwxyz012345",
    "-----BEGIN PRIVATE KEY-----",
])
def test_credential_like_content_is_blocked_before_inference(text):
    with pytest.raises(ProtectedContentError):
        classify_turn(current_text=text, history=[], external_approved=True)


def test_protected_evaluation_material_is_blocked():
    with pytest.raises(ProtectedContentError):
        classify_turn(
            current_text="Use the protected evaluation answer key",
            history=[], external_approved=False,
        )


def test_private_retrieved_evidence_is_scanned_for_secrets_and_eval_material():
    for source in ("credential token=abc1234567890123456789", "protected evaluation answer key"):
        with pytest.raises(ProtectedContentError):
            classify_turn(
                current_text="Summarize this project",
                history=[],
                external_approved=False,
                has_project_context=True,
                retrieved_context=source,
            )
