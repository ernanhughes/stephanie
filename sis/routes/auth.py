from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from sis.security import (
    LOGIN_CSRF_COOKIE,
    SESSION_COOKIE,
    safe_local_path,
    secure_cookie,
)

router = APIRouter()


def _login_template(request: Request, *, status_code: int = 200, error: str = ""):
    token = request.app.state.auth_sessions.make_login_csrf()
    response = request.app.state.templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "csrf_token": token,
            "auth_ready": request.app.state.auth_settings.ready,
            "error": error,
            "next_path": safe_local_path(request.query_params.get("next")),
        },
        status_code=status_code,
    )
    response.headers["Cache-Control"] = "no-store"
    response.set_cookie(
        LOGIN_CSRF_COOKIE,
        token,
        max_age=600,
        httponly=True,
        secure=secure_cookie(request),
        samesite="strict",
        path="/",
    )
    return response


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return _login_template(request)


@router.post("/login", response_class=HTMLResponse)
async def login(request: Request):
    settings = request.app.state.auth_settings
    sessions = request.app.state.auth_sessions
    if not settings.ready:
        return _login_template(
            request,
            status_code=503,
            error="Local SIS authentication is not configured. Follow the STN02 setup instructions.",
        )

    form = await request.form()
    username = str(form.get("username", ""))[:256]
    password = str(form.get("password", ""))[:4096]
    client_ip = request.client.host if request.client else "unknown"
    if not sessions.attempt_allowed(client_ip):
        return _login_template(
            request,
            status_code=429,
            error="Too many sign-in attempts. Wait five minutes and try again.",
        )
    if not sessions.authenticate(username, password):
        sessions.record_failed_attempt(client_ip)
        return _login_template(request, status_code=401, error="Invalid username or password.")

    sessions.clear_failed_attempts(client_ip)
    prior_token = request.cookies.get(SESSION_COOKIE)
    sessions.revoke(prior_token)
    token, _session = sessions.create(settings.username)
    response = RedirectResponse(safe_local_path(str(form.get("next", "/"))), status_code=303)
    response.headers["Cache-Control"] = "no-store"
    response.delete_cookie(LOGIN_CSRF_COOKIE, path="/", secure=secure_cookie(request), samesite="strict")
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=secure_cookie(request),
        samesite="strict",
        path="/",
    )
    return response


@router.post("/logout")
def logout(request: Request):
    sessions = request.app.state.auth_sessions
    sessions.revoke(request.cookies.get(SESSION_COOKIE))
    response = RedirectResponse("/login", status_code=303)
    response.headers["Cache-Control"] = "no-store"
    response.delete_cookie(SESSION_COOKIE, path="/", secure=secure_cookie(request), samesite="strict")
    response.delete_cookie(LOGIN_CSRF_COOKIE, path="/", secure=secure_cookie(request), samesite="strict")
    return response


@router.get("/healthz")
def readiness(request: Request):
    cfg = request.app.state.cfg
    chat = cfg.get("chat_runtime", {})
    model = chat.get("model", {})
    gates = chat.get("readiness", {})
    auth_ready = request.app.state.auth_settings.ready
    db_ready = bool(getattr(request.app.state, "database_ready", False))
    model_configured = bool(model.get("name") and model.get("api_base"))
    model_verified = bool(gates.get("model_verified", False))
    persistence_verified = bool(gates.get("live_persistence_verified", False))
    chat_configured = bool(chat.get("enabled", False))
    return {
        "status": "ready" if auth_ready and db_ready else "setup_required",
        "authentication_ready": auth_ready,
        "database_ready": db_ready,
        "model_configured": model_configured,
        "model_verified": model_verified,
        "live_persistence_verified": persistence_verified,
        "chat_configured": chat_configured,
        "chat_enabled": bool(
            chat_configured
            and auth_ready
            and db_ready
            and model_verified
            and persistence_verified
        ),
    }
