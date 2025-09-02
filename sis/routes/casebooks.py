from __future__ import annotations
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

router = APIRouter()


# -----------------
# Helpers
# -----------------
def _to_dict(x: Any) -> Dict:
    if x is None:
        return {}
    if hasattr(x, "to_dict"):
        try:
            return x.to_dict()  # type: ignore
        except Exception:
            pass
    out = {}
    for k in dir(x):
        if k.startswith("_"):
            continue
        try:
            v = getattr(x, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out


def _get_attr(x: Any, key: str, default=None):
    if isinstance(x, dict):
        return x.get(key, default)
    return getattr(x, key, default)


# -----------------
# Casebooks
# -----------------

@router.get("/casebooks", response_class=HTMLResponse)
def list_casebooks(
    request: Request,
    agent: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
    pipeline_run_id: Optional[int] = Query(default=None),
):
    """
    List casebooks with optional filters.
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    # Try ORM query for flexibility
    try:
        cb_cls = memory.casebooks.session.registry._class_registry.get("CaseBookORM")  # type: ignore
        q = memory.casebooks.session.query(cb_cls)
        if agent is not None:
            q = q.filter(cb_cls.agent_name == agent)  # type: ignore
        if tag is not None:
            q = q.filter(cb_cls.tag == tag)  # type: ignore
        if pipeline_run_id is not None:
            q = q.filter(cb_cls.pipeline_run_id == pipeline_run_id)  # type: ignore
        casebooks = q.order_by(getattr(cb_cls, "created_at", getattr(cb_cls, "id")) .desc()).all()  # type: ignore
    except Exception:
        # Fallback: no filters
        try:
            casebooks = memory.casebooks.session.query(memory.casebooks.session.registry._class_registry["CaseBookORM"]).all()  # type: ignore
        except Exception:
            casebooks = []

    # Enrich with counts if possible
    enriched = []
    for cb in casebooks:
        d = _to_dict(cb)
        try:
            cases = getattr(cb, "cases", None)
            if cases is not None:
                d["case_count"] = len(cases)
            else:
                CaseORM = memory.casebooks.session.registry._class_registry.get("CaseORM")  # type: ignore
                if CaseORM:
                    cnt = memory.casebooks.session.query(CaseORM).filter_by(casebook_id=_get_attr(cb, "id")).count()  # type: ignore
                    d["case_count"] = cnt
        except Exception:
            d.setdefault("case_count", None)
        enriched.append(d)

    return templates.TemplateResponse(
        "/casebooks/list.html",
        {
            "request": request,
            "casebooks": enriched,
            "filters": {"agent": agent, "tag": tag, "pipeline_run_id": pipeline_run_id},
        },
    )


@router.get("/casebooks/{casebook_id}", response_class=HTMLResponse)
def casebook_detail(
    request: Request,
    casebook_id: int,
    goal: Optional[str] = Query(default=None, description="Filter cases by goal_id"),
):
    """
    Show one casebook and its recent cases (optionally filtered by goal).
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    # Load casebook
    try:
        cb_cls = memory.casebooks.session.registry._class_registry["CaseBookORM"]  # type: ignore
        casebook = memory.casebooks.session.get(cb_cls, casebook_id)  # type: ignore
    except Exception:
        casebook = None
    if not casebook:
        return PlainTextResponse("Casebook not found", status_code=404)

    # Load cases
    try:
        CaseORM = memory.casebooks.session.registry._class_registry["CaseORM"]  # type: ignore
        q = memory.casebooks.session.query(CaseORM).filter_by(casebook_id=casebook_id)  # type: ignore
        if goal:
            q = q.filter(CaseORM.goal_id == goal)  # type: ignore
        cases = q.order_by(getattr(CaseORM, "created_at", getattr(CaseORM, "id")) .desc()).limit(200).all()  # type: ignore
    except Exception:
        cases = getattr(casebook, "cases", []) or []

    # Summaries
    goals = {}
    for c in cases:
        gid = _get_attr(c, "goal_id")
        goals[gid] = goals.get(gid, 0) + 1

    return templates.TemplateResponse(
        "/casebooks/detail.html",
        {
            "request": request,
            "casebook": _to_dict(casebook),
            "cases": [_to_dict(c) for c in cases],
            "goals": [{"goal_id": gid, "count": n} for gid, n in sorted(goals.items(), key=lambda kv: (-kv[1], kv[0] or ""))],
            "goal_filter": goal,
        },
    )


# -----------------
# Cases
# -----------------

@router.get("/cases", response_class=HTMLResponse)
def list_cases(
    request: Request,
    casebook_id: Optional[int] = Query(default=None),
    agent: Optional[str] = Query(default=None),
    goal: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """
    List recent cases, optionally filtered by casebook_id, agent_name, goal_id.
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    try:
        CaseORM = memory.casebooks.session.registry._class_registry["CaseORM"]  # type: ignore
        q = memory.casebooks.session.query(CaseORM)
        if casebook_id is not None:
            q = q.filter(CaseORM.casebook_id == casebook_id)  # type: ignore
        if agent is not None:
            q = q.filter(CaseORM.agent_name == agent)  # type: ignore
        if goal is not None:
            q = q.filter(CaseORM.goal_id == goal)  # type: ignore
        cases = q.order_by(getattr(CaseORM, "created_at", getattr(CaseORM, "id")) .desc()).limit(limit).all()  # type: ignore
    except Exception:
        cases = []

    return templates.TemplateResponse(
        "/cases/list.html",
        {
            "request": request,
            "cases": [_to_dict(c) for c in cases],
            "filters": {"casebook_id": casebook_id, "agent": agent, "goal": goal, "limit": limit},
        },
    )


@router.get("/cases/{case_id}", response_class=HTMLResponse)
def case_detail(request: Request, case_id: int):
    """
    Detailed case view:
      - header/meta
      - MARS summary cards (if present)
      - scorables table
      - raw JSON payloads (mars_summary, scores, meta) collapsible
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    try:
        CaseORM = memory.casebooks.session.registry._class_registry["CaseORM"]  # type: ignore
        case = memory.casebooks.session.get(CaseORM, case_id)  # type: ignore
    except Exception:
        case = None
    if not case:
        return PlainTextResponse("Case not found", status_code=404)

    case_d = _to_dict(case)

    # Scorables (role/rank/text/mars_confidence if present in meta)
    scorables = []
    try:
        scs = getattr(case, "scorables", []) or []
    except Exception:
        scs = []
    for s in scs:
        sd = _to_dict(s)
        meta = sd.get("meta") or {}
        if isinstance(meta, str):
            # if JSON serialized string
            try:
                import json
                meta = json.loads(meta)
            except Exception:
                meta = {"raw": sd.get("meta")}
        sd["meta"] = meta
        sd["text"] = meta.get("text")
        sd["mars_confidence"] = meta.get("mars_confidence")
        scorables.append(sd)

    # Try to lay out MARS summary as list of dimensions
    mars_cards = []
    msum = case_d.get("mars_summary") or {}
    if isinstance(msum, dict):
        for k, v in msum.items():
            # v is per-hypothesis? per-dimension? Accept both shapes
            if isinstance(v, dict) and ("agreement_score" in v or "dimension" in v):
                mars_cards.append({
                    "dimension": v.get("dimension", k),
                    "agreement_score": v.get("agreement_score"),
                    "std_dev": v.get("std_dev"),
                    "preferred_model": v.get("preferred_model"),
                    "primary_conflict": v.get("primary_conflict"),
                    "delta": v.get("delta"),
                    "explanation": v.get("explanation"),
                    "high_disagreement": v.get("high_disagreement"),
                })

    return templates.TemplateResponse(
        "/cases/detail.html",
        {
            "request": request,
            "case": case_d,
            "scorables": scorables,
            "mars_cards": [c for c in mars_cards if c.get("dimension")],
        },
    )
