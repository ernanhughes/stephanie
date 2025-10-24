from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from stephanie.jobs.overnight import OUT_DIR, run_once

router = APIRouter(prefix="/overnight", tags=["overnight"])

@router.get("", response_class=HTMLResponse)
async def dashboard(request: Request):
    templates = request.app.state.templates
    files = sorted(OUT_DIR.glob("*_suggestions.json"), reverse=True)
    latest = files[0] if files else None
    cards = json.loads(latest.read_text(encoding="utf-8")) if latest else []
    return templates.TemplateResponse("/overnight/dashboard.html", {"request": request, "cards": cards, "latest": latest.name if latest else None})

@router.post("/run", response_class=HTMLResponse)
async def run_now(request: Request):
    templates = request.app.state.templates
    cards = await run_once(request.app.state.container)
    return templates.TemplateResponse("/overnight/dashboard.html", {"request": request, "cards": cards, "latest": "just now"})
