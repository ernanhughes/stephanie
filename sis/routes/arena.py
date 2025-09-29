# sis/routes/tap.py
from __future__ import annotations
from typing import Optional
from fastapi import HTTPException

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter(prefix="/arena", tags=["arena"])


# --- DB page (list + detail) ------------------------------------------------
@router.get("", response_class=HTMLResponse)
def arena_page(request: Request):
    """
    DB-backed Arena view (list + detail from bus_events).
    """
    print("GET /arena page requested (DB mode)")
    templates = request.app.state.templates
    return templates.TemplateResponse("/arena/db.html", {"request": request})

@router.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("/arena/runs.html", {"request": request})

@router.get("/live", response_class=HTMLResponse)
def live_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("/arena/live.html", {"request": request})

# sis/routes/tap.py (only the changed/added bits shown)


def _store(request: Request):
    return request.app.state.memory.bus_events

# --- API (DB-backed) --------------------------------------------------------

@router.get("/api/runs")
def api_recent_runs(request: Request, limit: int = Query(50, ge=1, le=500)):
    store = _store(request)
    try:
        return store.recent_runs(limit=limit)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/api/run/{run_id}/events")
def api_run_events(request: Request, run_id: str, limit: int = Query(2000, ge=1, le=10000)):
    store = _store(request)
    try:
        return store.payloads_by_run(run_id, limit=limit)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/api/recent")
def api_recent(
    request: Request,
    limit: int = Query(200, ge=1, le=2000),
    run_id: Optional[str] = Query(None),
    subject_like: Optional[str] = Query(None),
    event: Optional[str] = Query(None),
    since_id: Optional[int] = Query(None),
):
    store = _store(request)
    try:
        if since_id:
            rows = store.since_id(since_id, limit=limit)
        else:
            rows = store.recent(limit=limit, run_id=run_id, subject_like=subject_like, event=event)
        return [r.to_dict(include_payload=False, include_extras=False) for r in rows]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/api/event/{event_id}")
def api_event_detail(request: Request, event_id: int):
    store = _store(request)
    row = store.get(event_id)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return row.to_dict(include_payload=True, include_extras=True)


@router.get("/api/events")
def api_events(request: Request, run_id: str):
    """
    All event bodies (payload_json) for a run; ascending time.
    """
    try:
        store = _store(request)
        bodies = store.payloads_by_run(run_id)
        return bodies
    except Exception as e:
        print("ERROR /arena/api/events:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


