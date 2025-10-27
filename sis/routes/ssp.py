# sis/routes/ssp.py
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from stephanie.utils.json_sanitize import sanitize

router = APIRouter(tags=["ssp"])

def _get_ssp(app):
    comp = getattr(app.state, "ssp", None)
    if comp is None:
        # optionally lazy-init here
        raise RuntimeError("SSP not initialized")
    return comp

@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return request.app.state.templates.TemplateResponse("ssp/dashboard.html", {"request": request})

@router.get("/status", response_class=JSONResponse)
def status(request: Request):
    try:
        ssp = _get_ssp(request.app)
        return JSONResponse(content=sanitize(ssp.status()))
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/start", response_class=JSONResponse)
def start(request: Request, max_steps: int | None = Query(default=None, ge=1)):
    ssp = _get_ssp(request.app)
    out = ssp.start(max_steps=max_steps)
    return JSONResponse(content=sanitize(out))

@router.post("/stop", response_class=JSONResponse)
def stop(request: Request):
    ssp = _get_ssp(request.app)
    out = ssp.stop()
    return JSONResponse(content=sanitize(out))

@router.post("/tick", response_class=JSONResponse)
def tick(request: Request):
    ssp = _get_ssp(request.app)
    out = ssp.tick() if hasattr(ssp, "tick") else ssp.jitter_tick()
    return JSONResponse(content=sanitize(out))
