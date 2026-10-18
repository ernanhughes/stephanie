"""Deterministic tests for the Stephanie -> Writer draft handoff (SWF01 phase 4).

These pin the contract: a real operation is invoked with the exact supported
arguments, Writer's verdict is recorded (never acted on), and every failure mode
produces an explicit ``unavailable``/``error`` record instead of an exception or a
silent fallback.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from stephanie.integrations import writer_handoff as wh


class FakeDiscovery:
    def __init__(self, exe: str | None = "C:/fake/writer.exe", error: Exception | None = None):
        self._exe = exe
        self._error = error

    def _resolve_executable(self) -> str:
        if self._error:
            raise self._error
        assert self._exe is not None
        return self._exe


@pytest.fixture()
def draft(tmp_path):
    path = tmp_path / "2506.21734_blog_20261018_190634.md"
    path.write_text("# Draft\n\nSome body text.\n", encoding="utf-8")
    return path


def _completed(payload=None, *, returncode=0, stdout=None, stderr=""):
    text = stdout if stdout is not None else (json.dumps(payload) if payload is not None else "")
    return SimpleNamespace(returncode=returncode, stdout=text, stderr=stderr)


PASS_PAYLOAD = {
    "id": "val_report_abc123",
    "target_type": "file",
    "status": "PASS",
    "decision": "ACCEPT",
    "gate": "PASS",
    "validation_version": "e1b7f66166df",
    "readiness": 100,
    "grade": "FREEZE_READY",
    "coverage": {"configured": 7, "successful": 7, "failed": 0},
    "total_findings": 0,
    "critical_count": 0,
    "high_count": 0,
    "medium_count": 0,
    "low_count": 0,
    "info_count": 0,
    "report_markdown": "# Validation Report\n\nStatus: PASS",
}

FAIL_PAYLOAD = {
    **PASS_PAYLOAD,
    "id": "val_report_def456",
    "status": "FAIL",
    "decision": "REJECT",
    "gate": "FAIL",
    "readiness": 42,
    "grade": "NOT_READY",
    "total_findings": 3,
    "critical_count": 1,
    "high_count": 2,
    "report_markdown": "# Validation Report\n\nStatus: FAIL",
}


def _handoff(monkeypatch, draft, *, discovery=None, record_dir=None, run=None):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(run, Exception):
            raise run
        if callable(run):
            return run(args, **kwargs)
        return run if run is not None else _completed(PASS_PAYLOAD)

    monkeypatch.setattr(wh.subprocess, "run", fake_run)
    handoff = wh.WriterDraftHandoff(
        cfg=wh.WriterRuntimeConfig(timeout_seconds=5),
        discovery=discovery or FakeDiscovery(),
        record_dir=record_dir,
    )
    return handoff, calls


def test_supported_operation_is_the_allowlisted_one():
    assert wh.WRITER_OPERATION == "validation.validate-file"
    assert ("validation", "validate-file") in wh.ALLOWED_OPERATIONS


def test_invokes_the_exact_supported_arguments(monkeypatch, draft):
    handoff, calls = _handoff(monkeypatch, draft)
    handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft), profile="nonfiction"))

    args, kwargs = calls[0]
    assert args == [
        "C:/fake/writer.exe",
        "validation",
        "validate-file",
        str(draft),
        "--profile",
        "nonfiction",
        "--json",
    ]
    assert kwargs["timeout"] == 5


def test_pass_records_writer_identity_and_no_review(monkeypatch, draft):
    handoff, _ = _handoff(monkeypatch, draft)
    record = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft)))

    assert record.status == "PASS"
    assert record.decision == "ACCEPT"
    assert record.gate == "PASS"
    assert record.writer_report_id == "val_report_abc123"
    assert record.writer_validation_version == "e1b7f66166df"
    assert record.readiness == 100
    assert record.grade == "FREEZE_READY"
    assert record.review_required is False
    assert record.human_review_reasons == []
    assert record.error is None
    assert record.artifact_sha256 and len(record.artifact_sha256) == 64
    assert record.artifact_bytes == draft.stat().st_size
    assert record.report_markdown.startswith("# Validation Report")


def test_fail_records_review_requirement(monkeypatch, draft):
    handoff, _ = _handoff(monkeypatch, draft, run=_completed(FAIL_PAYLOAD))
    record = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft)))

    assert record.status == "FAIL"
    assert record.decision == "REJECT"
    assert record.review_required is True
    assert record.finding_counts["critical"] == 1
    assert record.finding_counts["high"] == 2
    joined = " | ".join(record.human_review_reasons)
    assert "Writer status is FAIL" in joined
    assert "REJECT" in joined
    assert "critical" in joined


def test_record_is_persisted_next_to_the_artifact(monkeypatch, draft):
    handoff, _ = _handoff(monkeypatch, draft)
    record = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft)))

    expected = draft.parent / f"{draft.stem}.writer-handoff.json"
    assert record.record_path == str(expected)
    payload = json.loads(expected.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["writer_report_id"] == "val_report_abc123"
    assert payload["artifact_path"] == str(draft)


def test_record_dir_override(monkeypatch, draft, tmp_path):
    records = tmp_path / "records"
    handoff, _ = _handoff(monkeypatch, draft, record_dir=str(records))
    record = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft)))
    assert record.record_path == str(records / f"{draft.stem}.writer-handoff.json")


def test_correlation_id_is_used_or_generated(monkeypatch, draft):
    handoff, _ = _handoff(monkeypatch, draft)
    given = handoff.submit(
        wh.WriterHandoffRequest(artifact_path=str(draft), correlation_id="swh-fixed")
    )
    assert given.correlation_id == "swh-fixed"

    generated = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft)))
    assert generated.correlation_id.startswith("swh-")
    assert len(generated.correlation_id) == len("swh-") + 12


def test_writer_missing_is_unavailable_not_an_exception(monkeypatch, draft):
    discovery = FakeDiscovery(error=FileNotFoundError("no Writer executable"))
    handoff, calls = _handoff(monkeypatch, draft, discovery=discovery)
    record = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft)))

    assert calls == []  # nothing was invoked
    assert record.status == "unavailable"
    assert record.error == "writer executable not found"
    assert record.review_required is True
    assert record.writer_report_id is None


def test_nonzero_exit_is_an_explicit_error(monkeypatch, draft):
    handoff, _ = _handoff(
        monkeypatch, draft, run=_completed(returncode=3, stdout="", stderr="boom")
    )
    record = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft)))
    assert record.status == "error"
    assert record.error == "writer exited 3"
    assert record.detail == "boom"
    assert record.review_required is True


def test_timeout_is_an_explicit_error(monkeypatch, draft):
    import subprocess

    handoff, _ = _handoff(
        monkeypatch, draft, run=subprocess.TimeoutExpired(cmd="writer", timeout=5)
    )
    record = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft)))
    assert record.status == "error"
    assert record.error == "timeout"


def test_invalid_json_is_an_explicit_error(monkeypatch, draft):
    handoff, _ = _handoff(monkeypatch, draft, run=_completed(stdout="not json"))
    record = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft)))
    assert record.status == "error"
    assert record.error == "invalid JSON from writer"


def test_missing_artifact_is_an_explicit_error(monkeypatch, tmp_path):
    handoff, calls = _handoff(monkeypatch, tmp_path)
    record = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(tmp_path / "nope.md")))
    assert calls == []
    assert record.status == "error"
    assert record.error == "artifact not found"
    assert record.review_required is True


def test_never_auto_accepts_a_rejection(monkeypatch, draft):
    """The record reports the decision; it must not silently mark it accepted."""
    handoff, _ = _handoff(monkeypatch, draft, run=_completed(FAIL_PAYLOAD))
    record = handoff.submit(wh.WriterHandoffRequest(artifact_path=str(draft)))
    assert record.decision == "REJECT"
    assert record.review_required is True
    # Stephanie did not rewrite the artifact or the verdict.
    assert draft.read_text(encoding="utf-8").startswith("# Draft")
