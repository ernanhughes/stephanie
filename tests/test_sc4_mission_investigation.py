from __future__ import annotations

import json

import pytest

from stephanie.improvement.engine import ImprovementEngine
from stephanie.improvement.models import digest
from stephanie.improvement.sc3 import _sha, _worker_sources


def _snapshot():
    return {
        "episode_id": "episode-1",
        "project_key": "worker",
        "episode_title": "Worker response handling",
        "finding_id": "finding-1",
        "finding_title": "Empty result is not classified",
        "finding_kind": "defect",
        "finding_severity": "moderate",
        "finding_status": "open",
        "finding_rationale": "The result path does not distinguish an empty response from missing evidence.",
        "evidence_refs": ["worker:source:abc123"],
        "next_action": "Inspect the result classification path.",
    }


def test_mission_investigation_is_source_bound_deterministic_and_idempotent(tmp_path):
    engine = ImprovementEngine(tmp_path)
    first = engine.investigate_mission_observation(_snapshot())
    second = engine.investigate_mission_observation(_snapshot())

    assert first == second
    assert first["status"] == "completed"
    assert first["result"] == "REQUEST_ADDITIONAL_EVIDENCE"
    assert first["source_identity"] == "mission:episode/episode-1/finding/finding-1"
    assert first["hypothesis"]["generation"] == "deterministic_source_restating"
    assert first["hypothesis"]["evidence_ids"]
    assert first["candidate"] is None
    assert first["proposed_experiments"] == []
    assert first["authority"].endswith("no experiment or execution authority")
    assert len(json.loads((tmp_path / "investigations.json").read_text())) == 1


def test_mission_investigation_rejects_extra_or_unbounded_source_fields(tmp_path):
    engine = ImprovementEngine(tmp_path)
    with pytest.raises(ValueError, match="unsupported fields"):
        engine.investigate_mission_observation(_snapshot() | {"evaluator_answer": "correct"})
    with pytest.raises(ValueError, match="exceeds its bound"):
        engine.investigate_mission_observation(_snapshot() | {"finding_title": "x" * 301})


def test_exact_worker_source_observation_freezes_candidate_without_execution(tmp_path):
    source_digest = _sha(json.dumps(_worker_sources(), sort_keys=True).encode())
    worker_evidence_id = "worker-strategy-source-" + source_digest[:12]
    snapshot = _snapshot() | {"finding_kind": "worker_json_fence_normalization",
                             "evidence_refs": [worker_evidence_id]}
    investigation = ImprovementEngine(tmp_path).investigate_mission_observation(snapshot)

    assert investigation["result"] == "PROPOSE_DETERMINISTIC_FIXTURE_EXPERIMENT"
    assert investigation["candidate"]["candidate_id"]
    assert investigation["candidate"]["candidate_digest"]
    assert investigation["proposed_experiments"][0]["protocol_digest"]
    protocol = json.loads((tmp_path / "frozen-protocol.json").read_text())
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert protocol["mission_observation"]["source_identity"] == investigation["source_identity"]
    assert protocol["mission_observation"]["source_digest"] == investigation["source_digest"]
    assert protocol["candidate_digest"] == digest(candidate)
    assert set(candidate["evidence_ids"]) == {worker_evidence_id, "" + digest(investigation["source_identity"])[:24]}
    assert not (tmp_path / "worker-results.json").exists()
