from __future__ import annotations

import pytest
from fastapi import HTTPException

from sis.routes import stephanie as routes
from sis.services.context_handoff import StephanieContextHandoff
from sis.services.capsule_prompt_client import CapsulePromptClient


def _context(**overrides):
    value = {
        "event_id": "evt-1", "correlation_id": "corr-1", "application": "mission",
        "context_type": "repository", "subject": "writer", "revision": "a" * 40,
        "requested_mode": "generate_prompt",
    }
    value.update(overrides)
    return StephanieContextHandoff(**value)


def test_context_contract_is_versioned_bounded_and_rejects_unknown_fields():
    event = _context(selection_text="x" * 4000)
    assert event.schema_version == "stephanie.context.v1"
    with pytest.raises(ValueError):
        _context(selection_text="x" * 4001)
    with pytest.raises(ValueError):
        _context(unreviewed_payload="whole repository")
    assert _context(revision="working-tree").revision == "working-tree"


def test_document_context_without_a_bounded_snapshot_is_refused_before_capsule():
    request = routes.PreparePromptRequest(request_id="r1", context=_context(
        application="workbench", context_type="book.technical", subject="jev"))
    with pytest.raises(HTTPException) as exc:
        routes.prepare_prompt(request)
    assert exc.value.status_code == 422
    assert "no supported document snapshot" in exc.value.detail


def test_selection_envelope_must_match_prepared_document_text():
    doc={"schema_version":"stephanie.document-context.v1","document_id":"book-1",
        "document_type":"document","scope":"selection","content":"prepared passage",
        "content_fingerprint":"sha256:abc","document_revision":"sha256:revision",
        "document_state":"saved","classification":"public"}
    context=_context(application="workbench",context_type="selection",subject="book-1",
        document_type="document",scope="selection",content_fingerprint="sha256:abc",
        document_revision="sha256:revision",selection_text="different passage")
    request=routes.PreparePromptRequest(request_id="selection-mismatch",context=context,document_context=doc)
    with pytest.raises(HTTPException) as exc:
        routes.prepare_prompt(request)
    assert exc.value.status_code==422
    assert "selection text does not match" in exc.value.detail


def test_mission_must_supply_or_user_must_choose_a_revision():
    request = routes.PreparePromptRequest(request_id="missing-sha", context=_context(revision=None))
    with pytest.raises(HTTPException) as exc:
        routes.prepare_prompt(request)
    assert exc.value.status_code == 422
    assert "Select an exact commit or explicitly enter latest" in exc.value.detail


def test_prepare_prompt_is_idempotent_and_context_revision_bound(monkeypatch):
    routes._cache.clear()

    class FakeClient:
        calls = 0
        def subjects(self):
            return [{"id": "writer", "name": "Writer", "data_class": "internal"}]
        def processes(self):
            return [
                {"id": "repository.review", "version": "1.0", "context": "repository",
                 "status": "executable", "scopes": "repository", "document_types": ""},
                {"id": "repository.next", "version": "1.0", "context": "repository",
                 "status": "executable", "scopes": "repository", "document_types": ""},
                {"id": "repository.evaluate", "version": "1.0-design", "context": "repository",
                 "status": "design_only", "scopes": "repository", "document_types": ""},
            ]
        def prepare(self, *, subject, process_id, revision, instructions, destination, size_profile, token_budget):
            self.calls += 1
            return {"process": {"id": process_id, "version": "1"},
                    "destination": {"id": destination, "label": destination},
                    "size_profile": {"id": size_profile, "token_budget": 32000},
                    "stage": {"id": "substantive_review", "prompt_digest": "sha256:test"},
                    "subject": {"id": subject, "revision": revision},
                    "capsule": {"id": "cap-1", "digest": "sha256:cap"},
                    "context_measurement": {"estimated_projection_tokens": 30, "projection_budget_tokens": 32000, "estimated_rendered_prompt_tokens": 50, "incomplete_context": False},
                    "external_handoff_policy": {"classifications_present": ["internal"]},
                    "prompt": "Review this repository.", "dispatch": {"dispatched": False}}

    fake = FakeClient()
    monkeypatch.setattr(routes, "_client", lambda: fake)
    unconfirmed = routes.PreparePromptRequest(request_id="needs-confirmation", context=_context())
    with pytest.raises(HTTPException) as exc:
        routes.prepare_prompt(unconfirmed)
    assert exc.value.status_code == 422
    assert fake.calls == 0
    request = routes.PreparePromptRequest(request_id="stable-request", context=_context(), external_handoff_confirmed=True,
        destination="chatgpt_web", size_profile="standard")
    first = routes.prepare_prompt(request).body
    second = routes.prepare_prompt(request).body
    assert first == second
    assert fake.calls == 1
    assert b"Destination: chatgpt_web" in first
    assert b"not_dispatched" in first
    changed = routes.PreparePromptRequest(request_id="stable-request", context=_context(revision="b" * 40), external_handoff_confirmed=True,
        destination="codex", size_profile="standard")
    with pytest.raises(HTTPException) as exc:
        routes.prepare_prompt(changed)
    assert exc.value.status_code == 409
    with pytest.raises(ValueError):
        routes.PreparePromptRequest(request_id="design-only",context=_context(),
            operation="repository.evaluate",external_handoff_confirmed=True)


def test_capsule_child_environment_does_not_forward_database_or_provider_secrets(monkeypatch, tmp_path):
    (tmp_path / "src" / "capsule_app").mkdir(parents=True)
    monkeypatch.setenv("CAPSULE_DSN", "postgres://secret")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("PATH", "C:/Windows")
    env = CapsulePromptClient(root=tmp_path)._environment()
    assert "CAPSULE_DSN" not in env
    assert "OPENAI_API_KEY" not in env
    assert env["PATH"] == "C:/Windows"
