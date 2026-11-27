from __future__ import annotations

import re
from types import SimpleNamespace

from fastapi.testclient import TestClient

from sis.prompt_host import create_prompt_host
from sis.routes import stephanie as prompt_routes
from sis.security import AuthSettings, SessionStore, hash_password


def test_isolated_prompt_host_serves_existing_authenticated_prompt_route(monkeypatch):
    monkeypatch.setenv("STEPHANIE_CONTEXT_ORIGINS", "http://127.0.0.1:5173")
    app = create_prompt_host()
    settings = AuthSettings(
        username="codex-local",
        password_hash=hash_password("an isolated test password"),
        session_secret=b"test-only-local-session-secret-at-least-32-bytes",
        allowed_hosts=("127.0.0.1",),
        allowed_origins=("http://127.0.0.1:8766",),
    )
    app.state.auth_settings = settings
    app.state.auth_sessions = SessionStore(settings)
    monkeypatch.setattr(prompt_routes, "_client", lambda: SimpleNamespace(subjects=lambda: [
        {"id": "writer", "name": "Writer", "data_class": "internal"},
    ], processes=lambda: [{"id": "repository.review", "version": "1.0", "context": "repository",
                          "status": "executable", "scopes": "repository", "document_types": ""}]))

    with TestClient(app, base_url="http://127.0.0.1:8766") as client:
        health = client.get("/healthz")
        redirected = client.get("/stephanie", headers={"Accept": "text/html"}, follow_redirects=False)
        login = client.get("/login")
        csrf = re.search(r'name="_csrf" value="([^"]+)"', login.text).group(1)
        accepted = client.post("/login", data={
            "_csrf": csrf, "username": "codex-local", "password": "an isolated test password", "next": "/stephanie",
        }, headers={"Origin": "http://127.0.0.1:8766"}, follow_redirects=False)
        page = client.get("/stephanie")

    assert health.json() == {
        "status": "ready", "mode": "prompt-only", "authentication_ready": True,
        "database_required": False, "chat_enabled": False,
    }
    assert redirected.status_code == 303 and redirected.headers["location"] == "/login"
    assert login.headers["content-security-policy"] == "frame-ancestors 'self' http://127.0.0.1:5173"
    assert accepted.status_code == 303
    assert page.status_code == 200
    assert "Stephanie" in page.text and "repository.review" in page.text
    assert "readOnly = context.application === 'mission' && !!context.revision" in page.text
    assert "confirmBox.checked = false" in page.text
    assert "signal:controller.signal" in page.text
    assert "thisGeneration !== generation" in page.text
    assert "await navigator.clipboard.writeText(copiedResult.markdown)" in page.text
    assert "Clipboard write failed" in page.text
    assert "Select the rendered prompt and copy it manually" in page.text
    assert "catch (_) { status.textContent = 'Clipboard unavailable." not in page.text
    assert "result === copiedResult && generation === copiedGeneration" in page.text
