import json

import pytest

from stephanie.improvement.evolution02 import freeze
from stephanie.improvement.safety import ProtectedContextError, assert_model_context_safe, validate_reviewed_experience


@pytest.mark.parametrize("payload", [
    {"oracle_label": "VERIFIED"},
    {"hidden_expected_answer": "x"},
    {"evaluation_status": "PASS", "experiment_id": "current", "experiment_state": "evaluated"},
    {"visibility": "development_only"},
    {"integrity": "contaminated"},
])
def test_protected_or_untrusted_context_rejected(payload):
    with pytest.raises(ProtectedContextError):
        assert_model_context_safe(payload, current_experiment_id="current")


def test_reviewed_historical_experience_allowed():
    validate_reviewed_experience({"experience_id":"prior", "reuse_status":"OWNER_REVIEWED_HISTORICAL",
        "visibility":"MODEL_VISIBLE_HISTORY", "integrity":"TRUSTED_SANITIZED", "summary":"known prior result"})


def test_unreviewed_or_hidden_experience_rejected():
    with pytest.raises(ProtectedContextError):
        validate_reviewed_experience({"experience_id":"bad", "reuse_status":"UNREVIEWED",
            "visibility":"MODEL_VISIBLE_HISTORY", "integrity":"TRUSTED_SANITIZED"})


def test_freeze_is_idempotent_and_digest_protected(tmp_path):
    first = freeze(tmp_path)
    assert first["planned_generation_calls"] == 24
    assert freeze(tmp_path) == first
    altered = json.loads((tmp_path / "frozen-manifest.json").read_text())
    altered["sample_size_tasks"] = 9
    (tmp_path / "frozen-manifest.json").write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="differs"):
        freeze(tmp_path)
