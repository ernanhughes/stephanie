from __future__ import annotations

import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from sis.routes.auth import router as auth_router
from sis.security import (
    AuthSettings,
    SISAuthMiddleware,
    SessionStore,
    hash_password,
)
from sis.template_utils import SISTemplates


def make_secure_app(*, secure_cookies=False, session_ttl=3600):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    origin = "https://127.0.0.1:8000" if secure_cookies else "http://127.0.0.1:8000"
    settings = AuthSettings(
        username="local-user",
        password_hash=hash_password("a secure test password"),
        session_secret=b"test-only-session-secret-32-bytes-long",
        allowed_hosts=("127.0.0.1",),
        allowed_origins=(origin,),
        session_ttl_seconds=session_ttl,
        secure_cookies=secure_cookies,
    )
    app.state.auth_settings = settings
    app.state.auth_sessions = SessionStore(settings)
    app.state.templates = SISTemplates(directory="sis/templates")
    app.state.cfg = {
        "chat_runtime": {
            "enabled": False,
            "model": {"name": "fixture/model", "api_base": "http://model.invalid"},
            "readiness": {"model_verified": False, "live_persistence_verified": False},
        }
    }
    app.state.database_ready = True
    app.add_middleware(SISAuthMiddleware)
    app.include_router(auth_router)
    app.mount("/static", StaticFiles(directory="sis/static"), name="static")

    @app.get("/private")
    def private(request: Request):
        return {"username": request.state.username, "csrf_token": request.state.csrf_token}

    @app.post("/private/mutate")
    def mutate(request: Request):
        return {"updated": True, "username": request.state.username}

    return app


def _login_csrf(response) -> str:
    match = re.search(r'name="_csrf" value="([^"]+)"', response.text)
    assert match is not None
    return match.group(1)


def _sign_in(client, *, password="a secure test password", origin="http://127.0.0.1:8000"):
    page = client.get("/login")
    token = _login_csrf(page)
    return client.post(
        "/login",
        data={"_csrf": token, "username": "local-user", "password": password, "next": "/private"},
        headers={"Origin": origin},
        follow_redirects=False,
    )


def test_private_sis_routes_redirect_pages_and_reject_api_mutations_until_login():
    app = make_secure_app()
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        page = client.get("/private", headers={"Accept": "text/html"}, follow_redirects=False)
        api = client.post("/private/mutate", json={}, headers={"Origin": "http://127.0.0.1:8000"})
        static = client.get("/static/css/style.css")
        health = client.get("/healthz")

    assert page.status_code == 303
    assert page.headers["location"] == "/login"
    assert api.status_code == 401
    assert static.status_code == 200
    assert health.status_code == 200
    assert health.json()["chat_enabled"] is False


def test_login_rejects_bad_credentials_and_rotates_to_an_http_only_session():
    app = make_secure_app()
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        rejected = _sign_in(client, password="wrong password")
        assert rejected.status_code == 401
        assert "Invalid username or password" in rejected.text
        assert client.cookies.get("sis_session") is None

        accepted = _sign_in(client)
        assert accepted.status_code == 303
        assert accepted.headers["location"] == "/private"
        assert client.cookies.get("sis_session")
        cookie_headers = accepted.headers.get_list("set-cookie")
        session_cookie = next(value for value in cookie_headers if value.startswith("sis_session="))
        assert "HttpOnly" in session_cookie
        assert "SameSite=strict" in session_cookie
        assert "Secure" not in session_cookie  # plain loopback HTTP development
        assert client.get("/private").json()["username"] == "local-user"


def test_signing_in_again_invalidates_the_prior_session_token():
    app = make_secure_app()
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        assert _sign_in(client).status_code == 303
        old_token = client.cookies.get("sis_session")
        assert _sign_in(client).status_code == 303
        new_token = client.cookies.get("sis_session")
        assert new_token and new_token != old_token
        client.cookies.set("sis_session", old_token, domain="127.0.0.1", path="/")
        stale = client.get("/private", headers={"Accept": "application/json"})

    assert stale.status_code == 401


def test_csrf_host_and_origin_checks_protect_authenticated_mutations():
    app = make_secure_app()
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        assert _sign_in(client).status_code == 303
        csrf = client.get("/private").json()["csrf_token"]

        no_csrf = client.post(
            "/private/mutate", json={}, headers={"Origin": "http://127.0.0.1:8000"}
        )
        bad_csrf = client.post(
            "/private/mutate",
            json={},
            headers={"Origin": "http://127.0.0.1:8000", "X-CSRF-Token": "wrong"},
        )
        cross_origin = client.post(
            "/private/mutate",
            json={},
            headers={"Origin": "http://evil.example", "X-CSRF-Token": csrf},
        )
        untrusted_host = client.get("/private", headers={"Host": "rebind.evil.example"})
        accepted = client.post(
            "/private/mutate",
            json={},
            headers={"Origin": "http://127.0.0.1:8000", "X-CSRF-Token": csrf},
        )
        accepted_form = client.post(
            "/private/mutate",
            data={"_csrf": csrf},
            headers={"Origin": "http://127.0.0.1:8000"},
        )

    assert no_csrf.status_code == 403
    assert bad_csrf.status_code == 403
    assert cross_origin.status_code == 403
    assert untrusted_host.status_code == 400
    assert accepted.status_code == 200
    assert accepted_form.status_code == 200


def test_login_csrf_is_required_and_logout_revokes_the_server_side_session():
    app = make_secure_app()
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        page = client.get("/login")
        csrf = _login_csrf(page)
        no_csrf = client.post(
            "/login",
            data={"username": "local-user", "password": "a secure test password"},
            headers={"Origin": "http://127.0.0.1:8000"},
            follow_redirects=False,
        )
        assert no_csrf.status_code == 403

        login = client.post(
            "/login",
            data={"_csrf": csrf, "username": "local-user", "password": "a secure test password"},
            headers={"Origin": "http://127.0.0.1:8000"},
            follow_redirects=False,
        )
        assert login.status_code == 303
        csrf_token = client.get("/private").json()["csrf_token"]
        logout = client.post(
            "/logout",
            headers={"Origin": "http://127.0.0.1:8000", "X-CSRF-Token": csrf_token},
            follow_redirects=False,
        )
        after_logout = client.get("/private", headers={"Accept": "text/html"}, follow_redirects=False)

    assert logout.status_code == 303
    assert after_logout.status_code == 303


def test_https_login_sets_secure_session_cookie_and_expired_session_is_rejected():
    app = make_secure_app(secure_cookies=True, session_ttl=3600)
    with TestClient(app, base_url="https://127.0.0.1:8000") as client:
        login = _sign_in(client, origin="https://127.0.0.1:8000")
        session_cookie = next(
            value for value in login.headers.get_list("set-cookie")
            if value.startswith("sis_session=")
        )
        token = client.cookies.get("sis_session")
        key = app.state.auth_sessions._session_key(token)
        app.state.auth_sessions._sessions[key].expires_at = 0
        expired = client.get("/private", headers={"Accept": "application/json"})

    assert login.status_code == 303
    assert "Secure" in session_cookie
    assert expired.status_code == 401


def test_auth_not_configured_fails_closed_and_does_not_expose_private_routes():
    app = make_secure_app()
    app.state.auth_settings = AuthSettings(
        username="",
        password_hash="",
        session_secret=b"",
        allowed_hosts=("127.0.0.1",),
        allowed_origins=("http://127.0.0.1:8000",),
    )
    app.state.auth_sessions = SessionStore(app.state.auth_settings)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        private = client.get("/private")
        health = client.get("/healthz")
        login = client.get("/login")

    assert private.status_code == 503
    assert health.json()["authentication_ready"] is False
    assert login.status_code == 200
    assert "Authentication is not configured" in login.text


def test_explicit_remote_host_is_rejected_without_https():
    app = make_secure_app()
    settings = AuthSettings(
        username="local-user",
        password_hash=hash_password("a secure test password"),
        session_secret=b"remote-test-session-secret-at-least-32",
        allowed_hosts=("sis.example.test",),
        allowed_origins=("http://sis.example.test",),
    )
    app.state.auth_settings = settings
    app.state.auth_sessions = SessionStore(settings)
    with TestClient(app, base_url="http://sis.example.test") as client:
        response = client.get("/login")
    assert response.status_code == 400
    assert "requires HTTPS" in response.json()["detail"]
