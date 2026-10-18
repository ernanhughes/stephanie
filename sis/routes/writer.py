"""SIS routes for discovering Writer runtimes.

Discovery is read-only: it shells out to the Writer CLI with ``--json`` and
never executes a runtime. Execution stays disabled here and is governed by the
Writer service configuration.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_bridge(request: Request):
    """Return the configured Writer bridge, or None when not configured."""
    return getattr(request.app.state, "writer_bridge", None)


def _unavailable_reason(request: Request) -> str:
    cfg = getattr(request.app.state, "writer_cfg", None) or {}
    if not cfg:
        return "Writer integration is not configured."
    if not cfg.get("project_root"):
        return "Writer integration is missing a project_root."
    return "Writer integration is unavailable."


@router.get("/writer", response_class=HTMLResponse)
def writer_index(request: Request, q: str | None = Query(default=None)):
    templates = request.app.state.templates
    bridge = _get_bridge(request)
    results = []
    error = None
    status = None
    query = (q or "").strip()

    if bridge is None:
        error = _unavailable_reason(request)
    else:
        status = bridge.status()
        if query:
            payload = bridge.discovery.search(query)
            if payload.get("ok"):
                results = payload.get("results", [])
            else:
                error = payload.get("error", "search failed")

    return templates.TemplateResponse(
        "writer/index.html",
        {
            "request": request,
            "active_page": "writer",
            "query": query,
            "results": results,
            "error": error,
            "status": status,
        },
    )


@router.get("/writer/runtime/{runtime_name}", response_class=HTMLResponse)
def writer_runtime_detail(request: Request, runtime_name: str):
    templates = request.app.state.templates
    bridge = _get_bridge(request)
    describe = None
    operations = None
    error = None

    if bridge is None:
        error = _unavailable_reason(request)
    else:
        payload = bridge.discovery.describe(runtime_name)
        if payload.get("ok"):
            describe = payload
        else:
            error = payload.get("error", "describe failed")
        ops = bridge.discovery.operations(runtime_name)
        if ops.get("ok"):
            operations = ops.get("operations", [])

    return templates.TemplateResponse(
        "writer/runtime.html",
        {
            "request": request,
            "active_page": "writer",
            "runtime_name": runtime_name,
            "describe": describe,
            "operations": operations,
            "error": error,
        },
    )
