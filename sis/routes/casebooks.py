from __future__ import annotations
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse

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


@router.get("/casebooks", response_class=HTMLResponse)
def list_casebooks(
    request: Request,
    agent: Optional[str] = Query(default=None, description="Filter by agent_name"),
    tag: Optional[str] = Query(default=None, description="Filter by casebook tag"),
    pipeline_run_id: Optional[int] = Query(default=None, description="Filter by pipeline_run_id"),
    limit: int = Query(default=200, ge=1, le=1000),
):
    memory = request.app.state.memory
    templates = request.app.state.templates

    # 1) Pull from the store with filters
    casebooks = memory.casebooks.list_casebooks(
        agent_name=agent,
        tag=tag,
        pipeline_run_id=pipeline_run_id,
        limit=limit,
    )

    # 2) Convert to dicts for the template (include case_count w/o heavy joins)
    items = []
    for cb in casebooks:
        d = cb.to_dict(include_cases=False, include_counts=True)
        items.append(d)

    # 3) Render
    return templates.TemplateResponse(
        "/casebooks/list.html",
        {
            "request": request,
            "casebooks": items,
            "filters": {
                "agent": agent,
                "tag": tag,
                "pipeline_run_id": pipeline_run_id,
                "limit": limit,
            },
        },
    )


def _count_by_goal(cases) -> list[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for c in cases:
        gid = getattr(c, "goal_id", None) if not isinstance(c, dict) else c.get("goal_id")
        counts[gid] = counts.get(gid, 0) + 1
    return [{"goal_id": gid, "count": n} for gid, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0] or ""))]

@router.get("/casebooks/{run_id}", response_class=HTMLResponse)
def casebook_for_run(
    request: Request,
    run_id: int,
    goal: Optional[str] = Query(default=None, description="Filter cases by goal_id"),
    limit: int = Query(default=500, ge=1, le=2000),
):
    """
    Show the casebook **for a pipeline run** and its recent cases.
    This matches links like: <a href="/casebooks/{{ run.id }}">.
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    cb = memory.casebooks.get_for_run_id(run_id)
    if not cb:
        return PlainTextResponse("Casebook not found for run", status_code=404)

    # Fetch cases via the store with optional goal filter
    cases = memory.casebooks.list_cases(
        casebook_id=cb.id,
        goal_id=goal,
        limit=limit,
    )

    return templates.TemplateResponse(
        "/casebooks/detail.html",
        {
            "request": request,
            "casebook": cb.to_dict(include_cases=False, include_counts=True),
            "cases": [c.to_dict(include_scorables=False) for c in cases],
            "goals": _count_by_goal(cases),
            "goal_filter": goal,
        },
    )

@router.get("/casebooks/id/{casebook_id}", response_class=HTMLResponse)
def casebook_detail_by_id(
    request: Request,
    casebook_id: int,
    goal: Optional[str] = Query(default=None, description="Filter cases by goal_id"),
    limit: int = Query(default=500, ge=1, le=2000),
):
    memory = request.app.state.memory
    templates = request.app.state.templates

    cb = memory.casebooks.get_casebook(casebook_id)
    if not cb:
        return PlainTextResponse("Casebook not found", status_code=404)

    cases = memory.casebooks.list_cases(
        casebook_id=cb.id,
        goal_id=goal,
        limit=limit,
    )

    return templates.TemplateResponse(
        "/casebooks/detail.html",
        {
            "request": request,
            "casebook": cb.to_dict(include_cases=False, include_counts=True),
            "cases": [c.to_dict(include_scorables=False) for c in cases],
            "goals": _count_by_goal(cases),
            "goal_filter": goal,
        },
    )


@router.get("/casebooks/{casebook_id}", response_class=HTMLResponse)
def casebook_detail(
    request: Request,
    run_id: int,
    goal: Optional[str] = Query(default=None, description="Filter cases by goal_id"),
):
    """
    Show one casebook and its recent cases (optionally filtered by goal).
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    try:
        casebook = memory.casebooks.get_for_run_id(run_id)  # type: ignore
    except Exception:
        casebook = None
    if not casebook:
        return PlainTextResponse("Casebook not found", status_code=404)

    # Load cases
    try:
        cases = memory.casebooks.get_cases_for_casebook(casebook_id=casebook.id)  # type: ignore
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

    cases = memory.casebooks.list_cases(
        casebook_id=casebook_id,
        agent_name=agent,
        goal_id=goal,
        limit=limit,
    )

    return templates.TemplateResponse(
        "/cases/list.html",
        {
            "request": request,
            "cases": [c.to_dict(include_scorables=False) for c in cases],
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

    case = memory.casebooks.get_case_by_id(case_id)
    if not case:
        return PlainTextResponse("Case not found", status_code=404)

    case_d = case.to_dict(include_scorables=False)

    # Scorables (role/rank/text/mars_confidence if present in meta)
    scorables = []
    for s in (getattr(case, "scorables", []) or []):
        sd = s.to_dict()
        meta = sd.get("meta") or {}
        # meta should already be a dict (SA_JSON), but be defensive:
        if isinstance(meta, str):
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


@router.get("/casebooks/{case_id}/uncertain", response_class=HTMLResponse)
def list_uncertain(request: Request, case_id: int):
    memory = request.app.state.memory
    rows = memory.casebooks.list_scorables_by_role(case_id, role="uncertain_candidate", limit=500)
    # If you don't want a template yet, just return a quick HTML:
    items = "".join(
        f"""
        <div style="border:1px solid #ddd; padding:10px; margin:8px;">
          <div><b>Agent:</b> {r.meta.get('agent_name')}</div>
          <div><b>Section:</b> {r.meta.get('section_name')}</div>
          <div><b>Scores:</b> K={r.meta.get('knowledge_score'):.2f} AI={r.meta.get('ai_judge_score'):.2f} Q={r.meta.get('artifact_quality'):.2f} Blend={r.meta.get('blended_score'):.2f}</div>
          <pre style="white-space:pre-wrap;">{(r.text or '').strip()[:4000]}</pre>
          <form method="post" action="/casebooks/{case_id}/label/{r.id}">
            <label>Human star (-5..5):</label>
            <input name="human_star" type="number" min="-5" max="5" step="1" required>
            <button>Save</button>
          </form>
        </div>
        """
        for r in rows
    )
    return HTMLResponse(f"<h2>Borderline candidates for labeling (case {case_id})</h2>{items or '<p>None</p>'}")

@router.post("/casebooks/{case_id}/label/{scorable_id}")
def label_candidate(request: Request, case_id: int, scorable_id: int, human_star: int = Form(...)):
    memory = request.app.state.memory
    memory.casebooks.update_scorable_meta(scorable_id, {"human_star": int(human_star)})
    return RedirectResponse(url=f"/casebooks/{case_id}/uncertain", status_code=303)
