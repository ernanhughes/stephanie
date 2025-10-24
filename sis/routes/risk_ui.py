from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse

from stephanie.components.gap.risk.orchestrator import GapRiskOrchestrator

router = APIRouter(prefix="/risk", tags=["risk-ui"])


def _orch(request: Request, profile: Optional[str] = None) -> GapRiskOrchestrator:
    # Cache a default orchestrator in app.state; allow per-call profile override
    key = f"gap_risk_orchestrator::{profile or 'chat.standard'}"
    orch = getattr(request.app.state, key, None)
    if not orch:
        orch = GapRiskOrchestrator(
            request.app.state.container,
            policy_profile=profile or "chat.standard",
        )
        setattr(request.app.state, key, orch)
    return orch


@router.get("", response_class=HTMLResponse)
async def form_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("/risk/form.html", {"request": request})


@router.post("/analyze", response_class=HTMLResponse)
async def analyze_form(
    request: Request,
    goal: str = Form(...),
    reply: str = Form(...),
    model_alias: str = Form("chat-hrm"),
    monitor_alias: str = Form("tiny-monitor"),
    policy_profile: Optional[str] = Form(None),
):
    templates = request.app.state.templates
    orch = _orch(request, policy_profile)
    rec = await orch.evaluate(
        goal=goal,
        reply=reply,
        model_alias=model_alias,
        monitor_alias=monitor_alias,
    )
    return templates.TemplateResponse("/risk/result.html", {"request": request, "rec": rec})


@router.post("/api/analyze")
async def analyze_api(
    request: Request,
    goal: str = Form(...),
    reply: str = Form(...),
    model_alias: str = Form("chat-hrm"),
    monitor_alias: str = Form("tiny-monitor"),
    policy_profile: Optional[str] = Form(None),
):
    orch = _orch(request, policy_profile)
    rec = await orch.evaluate(
        goal=goal,
        reply=reply,
        model_alias=model_alias,
        monitor_alias=monitor_alias,
    )
    return JSONResponse(rec)
