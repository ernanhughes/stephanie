# sis/routes/cards.py
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/cards", tags=["cards"])

def _store(request: Request):
    mem = request.app.state.memory
    if not getattr(mem, "sis_cards", None):
        raise RuntimeError("memory.sis_cards not configured")
    return mem.sis_cards

@router.get("", response_class=HTMLResponse)
def cards_page(request: Request, key: Optional[str] = None, scope: str = "arena"):
    templates = request.app.state.templates
    return templates.TemplateResponse("/cards/list.html", {"request": request, "key": key, "scope": scope})

@router.get("/api/recent")
def api_recent(request: Request, scope: Optional[str] = Query(None), limit: int = Query(100, ge=1, le=1000)):
    store = _store(request)
    print("api_recent: fetching recent cards", {"scope": scope, "limit": limit})
    rows = store.recent()
    print(f"api_recent: found {len(rows)} rows")
    return [r.to_dict(include_cards=True) for r in rows]

@router.get("/api/key/{key}")
def api_by_key(request: Request, key: str, limit: int = Query(50, ge=1, le=500)):
    store = _store(request)
    rows = store.by_key(key, limit=limit)
    return [r.to_dict(include_cards=True) for r in rows]
