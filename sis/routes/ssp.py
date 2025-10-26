# sis/routes/ssp.py
# sis/routes/ssp.py
from __future__ import annotations
import threading
import traceback
from typing import Any, Dict, Optional
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from stephanie.components.ssp.component import SSPComponent
from stephanie.utils.json_sanitize import sanitize, dumps_safe

# We keep API and UI under one router for simplicity
router = APIRouter()

def _ensure_ssp(app) -> SSPComponent:
    # lazy-init and cache the component on app.state
    if not hasattr(app.state, "ssp_component") or app.state.ssp_component is None:
        from stephanie.components.ssp.component import SSPComponent
        cfg = getattr(app.state, "config", {})
        logger = getattr(app.state, "logger", None)
        event_publisher = getattr(getattr(app.state, "container", None), "event_publisher", None)
        app.state.ssp_component = SSPComponent(cfg=cfg, logger=logger, event_publisher=event_publisher)
    return app.state.ssp_component

# -------- UI --------

@router.get("/ssp", response_class=HTMLResponse)
def ssp_index(request: Request):
    templates = request.app.state.templates
    ssp = _ensure_ssp(request.app)
    return templates.TemplateResponse(
        "ssp/index.html",
        {
            "request": request,
            "boot": sanitize(ssp.status()),
        },
    )

# -------- JSON API --------

@router.get("/v1/ssp/status", response_class=JSONResponse)
def ssp_status(request: Request):
    ssp = _ensure_ssp(request.app)
    return JSONResponse(content=sanitize(ssp.status()))

@router.post("/v1/ssp/start", response_class=JSONResponse)
def ssp_start(request: Request, max_steps: Optional[int] = Query(default=None, ge=1)):
    ssp = _ensure_ssp(request.app)
    out = ssp.start(max_steps=max_steps)
    return JSONResponse(content=sanitize(out))

@router.post("/v1/ssp/stop", response_class=JSONResponse)
def ssp_stop(request: Request):
    ssp = _ensure_ssp(request.app)
    out = ssp.stop()
    return JSONResponse(content=sanitize(out))

@router.post("/v1/ssp/tick", response_class=JSONResponse)
def ssp_tick(request: Request):
    ssp = _ensure_ssp(request.app)
    out = ssp.tick()
    return JSONResponse(content=sanitize(out))

# ---- helpers ---------------------------------------------------------------

def _get_ssp(app) -> Any:
    """
    Lazily construct and cache the SSP component on app.state.
    Expects your component at stephanie.components.ssp.component.SSPComponent
    and its public methods: start(max_steps: Optional[int]), stop(), status(), tick().
    """
    comp = getattr(app.state, "ssp_component", None)
    if comp is None:
        from stephanie.components.ssp.component import SSPComponent
        comp = SSPComponent(cfg=app.state.config, memory=app.state.memory, logger=getattr(app.state, "logger", None))
        app.state.ssp_component = comp
    return comp

def _thread_running(app) -> bool:
    t = getattr(app.state, "ssp_thread", None)
    return bool(t and t.is_alive())

def _ok(data: Dict[str, Any]) -> JSONResponse:
    return JSONResponse(sanitize(data))

def _err(msg: str, exc: Optional[BaseException] = None) -> JSONResponse:
    detail = {"error": msg}
    if exc is not None:
        detail["traceback"] = traceback.format_exc()
    return JSONResponse(sanitize(detail), status_code=500)

# ---- pages ----------------------------------------------------------------

@router.get("", response_class=HTMLResponse)
def dashboard(request: Request):
    """
    Minimal HTML dashboard for SSP control + live status.
    """
    templates = request.app.state.templates
    return templates.TemplateResponse("/ssp/dashboard.html", {"request": request})

# ---- API ------------------------------------------------------------------

@router.get("/status")
def status(request: Request):
    try:
        comp = _get_ssp(request.app)
        st = comp.get_status() if hasattr(comp, "get_status") else comp.status()
        return _ok({"thread": _thread_running(request.app), "status": st})
    except Exception as e:
        return _err("Failed to fetch status", e)

@router.post("/start")
def start(request: Request, max_steps: Optional[int] = None):
    try:
        comp = _get_ssp(request.app)
        if _thread_running(request.app):
            return _ok({"message": "SSP already running", "status": comp.get_status() if hasattr(comp, "get_status") else comp.status()})

        def _runner():
            try:
                comp.start(max_steps=max_steps)
            except Exception:
                # Do not crash the server; expose errors via status/logs
                pass

        t = threading.Thread(target=_runner, name="ssp-thread", daemon=True)
        t.start()
        request.app.state.ssp_thread = t
        return _ok({"message": "SSP started", "thread": True})
    except Exception as e:
        return _err("Failed to start SSP", e)

@router.post("/stop")
def stop(request: Request):
    try:
        comp = _get_ssp(request.app)
        if hasattr(comp, "stop"):
            comp.stop()
        return _ok({"message": "Stop signal sent"})
    except Exception as e:
        return _err("Failed to stop SSP", e)

@router.post("/tick")
def tick(request: Request):
    """
    Manual jitter bridge tick for substrate feed. Returns sanitized payload.
    """
    try:
        comp = _get_ssp(request.app)
        if not getattr(comp, "is_running", False):
            # if component exposes boolean flag; otherwise ignore
            pass
        out = None
        if hasattr(comp, "jitter_tick"):
            out = comp.jitter_tick()
        elif hasattr(comp, "tick"):
            out = comp.tick()
        else:
            return _err("Component has no jitter_tick() / tick()")
        return JSONResponse(dumps_safe(out), media_type="application/json")
    except Exception as e:
        return _err("Failed to tick SSP", e)
