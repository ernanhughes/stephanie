# sis/routes/ssp_ui.py
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse
from stephanie.utils.json_sanitize import dumps_safe, sanitize

router = APIRouter()

@router.post("/ssp/start", response_class=JSONResponse)
def ssp_start(request: Request, steps: Optional[int] = Query(default=None)):
    ssp = request.app.state.ssp
    if ssp is None:
        return JSONResponse({"ok": False, "error": "SSP not available"}, status_code=500)
    ssp.start(max_steps=steps, background=(steps is None))
    return JSONResponse({"ok": True, "event": "started", "steps": steps})

@router.post("/ssp/tick", response_class=JSONResponse)
def ssp_tick(request: Request):
    ssp = request.app.state.ssp
    if ssp is None:
        return JSONResponse({"ok": False, "error": "SSP not available"}, status_code=500)
    out = ssp.tick()
    return JSONResponse(sanitize(out))

@router.get("/ssp/status", response_class=JSONResponse)
def ssp_status(request: Request):
    ssp = request.app.state.ssp
    if ssp is None:
        return JSONResponse({"status": "unavailable"})
    return JSONResponse(sanitize(ssp.status()))

@router.get("/ssp/results", response_class=HTMLResponse)
def ssp_results(request: Request, limit: int = Query(50, ge=1, le=500)):
    """
    Show recent verified episodes as 'cards' (proposal → answer → score).
    """
    templates = request.app.state.templates
    memory = request.app.state.memory
    # Minimal store: use plan_traces and filter recent trainer/solver/verifier triples
    rows = memory.plan_traces.list_recent(kind="ssp", limit=limit)  # implement a small helper in MemoryTool
    return templates.TemplateResponse(
        "/ssp/results.html",
        {"request": request, "rows": rows}
    )
