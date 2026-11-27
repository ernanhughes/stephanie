"""Isolated host for deterministic Capsule prompt handoff.

This host mounts the existing authenticated Stephanie prompt UI and route only.
It deliberately does not initialize MemoryTool, chat, Worker, or database state,
so local interface acceptance can run without activating those services.
"""
from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from sis.routes import auth as auth_routes
from sis.routes import stephanie as stephanie_routes
from sis.security import SISAuthMiddleware, SessionStore, load_auth_settings
from sis.template_utils import SISTemplates

ROOT = Path(__file__).resolve().parent.parent


def create_prompt_host() -> FastAPI:
    app = FastAPI(title="Stephanie Prompt Handoff", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.auth_settings = load_auth_settings()
    app.state.auth_sessions = SessionStore(app.state.auth_settings)
    app.state.templates = SISTemplates(directory=str(ROOT / "sis" / "templates"))
    app.state.cfg = {"chat_runtime": {"enabled": False}}

    app.add_middleware(SISAuthMiddleware)
    app.mount("/static", StaticFiles(directory=str(ROOT / "sis" / "static")), name="static")
    app.add_api_route("/login", auth_routes.login_page, methods=["GET"])
    app.add_api_route("/login", auth_routes.login, methods=["POST"])
    app.add_api_route("/logout", auth_routes.logout, methods=["POST"])
    app.include_router(stephanie_routes.router)

    @app.get("/")
    def home():
        return RedirectResponse("/stephanie", status_code=303)

    @app.get("/healthz")
    @app.get("/health")
    def health():
        ready = bool(app.state.auth_settings.ready)
        return JSONResponse({
            "status": "ready" if ready else "setup_required",
            "mode": "prompt-only",
            "authentication_ready": ready,
            "database_required": False,
            "chat_enabled": False,
        }, status_code=200 if ready else 503)

    @app.get("/{unmatched:path}")
    def unsupported_route(unmatched: str):
        return RedirectResponse("/stephanie", status_code=303)

    return app


app = create_prompt_host()


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("SIS_HOST", "127.0.0.1"),
                port=int(os.environ.get("SIS_PORT", "8766")), reload=False)
