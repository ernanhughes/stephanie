# sis/routes/arena.py
from __future__ import annotations
import json
import time
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
import logging

_logger = logging.getLogger(__name__)

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


# stephanie/routes/tap.py
from fastapi import HTTPException

@router.get("/api/provenance/{case_id}")
async def api_provenance(request: Request, case_id: str):
    """Get full provenance chain for a case"""
    store = request.app.state.memory
    
    try:
        # Get case
        case = store.casebooks.get_case_by_id(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get casebook
        casebook = store.casebooks.get_casebook(case.casebook_id)
        
        # Get scorables
        scorables = store.casebooks.list_scorables(case_id)
        
        # Get chat (if linked)
        chat = None
        try:
            chat_id = case.meta.get("chat_id") if case.meta else None
            if chat_id:
                chat = store.chats.get_chat(chat_id)
        except Exception:
            pass
        
        # Get paper
        paper = None
        try:
            paper_id = case.meta.get("paper_id") if case.meta else None
            if paper_id:
                paper = store.documents.get_document(paper_id)
        except Exception:
            pass
        
        # Extract provenance chain
        provenance_chain = []
        
        # 1. Paper
        if paper:
            provenance_chain.append({
                "type": "paper",
                "id": paper.id,
                "title": paper.title,
                "url": f"/papers/{paper.id}",
                "meta": {
                    "authors": getattr(paper, "authors", []),
                    "year": getattr(paper, "year", None),
                    "domain": getattr(paper, "domain", "general")
                }
            })
        
        # 2. CaseBook
        if casebook:
            provenance_chain.append({
                "type": "casebook",
                "id": casebook.id,
                "name": casebook.name,
                "url": f"/casebooks/{casebook.id}",
                "meta": {
                    "description": casebook.description,
                    "tags": casebook.tags,
                    "created_at": casebook.created_at.isoformat() if casebook.created_at else None
                }
            })
        
        # 3. Case
        provenance_chain.append({
            "type": "case",
            "id": case.id,
            "prompt": case.prompt_text[:200] + "..." if len(case.prompt_text or "") > 200 else case.prompt_text or "",
            "url": f"/cases/{case.id}",
            "meta": {
                "goal_id": case.goal_id,
                "agent_name": case.agent_name,
                "created_at": case.created_at.isoformat() if case.created_at else None
            }
        })
        
        # 4. Scorables
        scorable_items = []
        for s in scorables or []:
            scorable_items.append({
                "type": "scorable",
                "id": s.id,
                "role": s.role,
                "text": (s.text or "")[:200] + "..." if len(s.text or "") > 200 else s.text,
                "url": f"/scorables/{s.id}",
                "meta": {
                    "pipeline_run_id": s.pipeline_run_id,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "score": getattr(s, "score", None),
                    "attributes": s.meta or {}
                }
            })
        
        # 5. Chat (if available)
        if chat:
            provenance_chain.append({
                "type": "chat",
                "id": chat.id,
                "title": chat.title,
                "url": f"/chats/{chat.id}",
                "meta": {
                    "provider": chat.provider,
                    "created_at": chat.created_at.isoformat() if chat.created_at else None,
                    "message_count": len(chat.messages) if hasattr(chat, "messages") else 0
                }
            })
        
        # Get supporting knowledge elements
        supports = []
        for s in scorables or []:
            if s.role == "arena_citations":
                try:
                    data = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                    for citation in data.get("citations", [])[:5]:
                        supports.append({
                            "text": (citation.get("support", {}).get("text") or "")[:100] + "..." if len(citation.get("support", {}).get("text") or "") > 100 else citation.get("support", {}).get("text"),
                            "origin": citation.get("support", {}).get("origin"),
                            "variant": citation.get("support", {}).get("variant"),
                            "similarity": citation.get("similarity", 0.0),
                            "url": f"/scorables/{citation.get('support', {}).get('id')}" if citation.get("support", {}).get("id") else None
                        })
                except Exception:
                    pass
        
        # Get metrics
        metrics_scorable = None
        for s in scorables or []:
            if s.role == "metrics":
                metrics_scorable = s
                break
        
        metrics = {}
        if metrics_scorable:
            try:
                metrics = json.loads(metrics_scorable.text) if isinstance(metrics_scorable.text, str) else (metrics_scorable.text or {})
            except Exception:
                pass
        
        return {
            "case": {
                "id": case.id,
                "prompt": case.prompt_text[:200] + "..." if len(case.prompt_text or "") > 200 else case.prompt_text or "",
                "agent_name": case.agent_name,
                "goal_id": case.goal_id,
                "created_at": case.created_at.isoformat() if case.created_at else None,
                "meta": case.meta or {}
            },
            "provenance_chain": provenance_chain,
            "scorables": scorable_items,
            "supports": supports,
            "metrics": metrics,
            "actions": {
                "rescore_url": f"/api/scorables/{case.id}/rescore",
                "edit_url": f"/cases/{case.id}/edit",
                "view_full_url": f"/cases/{case.id}"
            }
        }
        
    except Exception as e:
        _logger.error(f"Provenance lookup failed for case {case_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Provenance lookup failed: {str(e)}")

@router.post("/api/scorables/{case_id}/rescore")
async def api_rescore(request: Request, case_id: str):
    """Rescore a case and update metrics"""
    store = request.app.state.memory
    
    try:
        # Get case
        case = store.casebooks.get_case_by_id(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        

        # Trigger rescoring (this would call your UniversalScorerAgent or similar)
        from stephanie.agents.scorable_loader_agent import ScorableLoaderAgent
        
        # Create agent instance
        agent = ScorableLoaderAgent(
            cfg=request.app.state.cfg,
            memory=store,
            container=request.app.state.container,
            logger=request.app.state.logger
        )
        
        # Run rescoring
        context = {
            "case_id": case_id,
            "pipeline_run_id": case.pipeline_run_id,
            "rescore_requested": True
        }
        
        result = await agent.run(context)
        
        # Emit score update event
        await store.bus.publish("scorables.score_updated", {
            "case_id": case_id,
            "new_metrics": result.get("metrics", {}),
            "timestamp": time.time()
        })
        
        return {
            "status": "success",
            "case_id": case_id,
            "new_metrics": result.get("metrics", {}),
            "message": "Case rescored successfully"
        }
        
    except Exception as e:
        _logger.error(f"Rescoring failed for case {case_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rescoring failed: {str(e)}")

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


