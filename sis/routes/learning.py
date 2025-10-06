# sis/routes/learning.py
from __future__ import annotations
import time
from typing import Optional
from fastapi import HTTPException
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
import logging

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/arena/learning", tags=["learning"])

def _store(request: Request):
    """Get BusEventStore instance from app state"""
    return request.app.state.memory.bus_events

def _run_store(request: Request):
    """Get PipelineRunStore instance from app state"""
    return request.app.state.memory.pipeline_runs

# --- Learning Views ---------------------------------------------------------

@router.get("", response_class=HTMLResponse)
def learning_page(request: Request):
    """
    Main learning visualization page showing knowledge transfer evidence.
    """
    _logger.info("GET /arena/learning requested")
    templates = request.app.state.templates
    return templates.TemplateResponse("/arena/learning.html", {"request": request})

@router.get("/run/{run_id}", response_class=HTMLResponse)
def learning_run_detail(request: Request, run_id: str):
    """
    Dedicated learning view for a specific pipeline run.
    """
    _logger.info(f"GET /arena/learning/run/{run_id} requested")
    
    # Get run details
    run_store = _run_store(request)
    run = run_store.get_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    # Get learning evidence for this specific run
    store = _store(request)
    rows = store.payloads_by_run(run_id, limit=2000)
    
    # Build transfer matrix
    from stephanie.tools.evidence_extractor import build_transfer_matrix_from_rows
    matrix = build_transfer_matrix_from_rows(rows)
    
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "/arena/learning_run.html", 
        {
            "request": request,
            "run": run,
            "matrix": matrix
        }
    )

# --- Learning API Endpoints -------------------------------------------------

@router.get("/api/transfer")
def api_learning_transfer(
    request: Request,
    run_id: Optional[str] = Query(None),
    agent: Optional[str] = Query(None),
    section: Optional[str] = Query(None),
    limit: int = Query(2000, ge=1, le=10000)
):
    """
    Build a knowledge transfer matrix from event logs.
    Query params:
      - run_id (optional): restrict to a single run
      - agent (optional)
      - section (optional)
      - limit (optional): max events per run (default 2000)
    """
    try:
        store = _store(request)
        rows = []
        
        if run_id:
            rows = store.payloads_by_run(run_id, limit=limit)
        else:
            # Combine the latest N runs to show "global" transfer
            recent_runs = store.recent_runs(limit=20)
            for r in recent_runs:
                rows.extend(store.payloads_by_run(r["run_id"], limit=limit))
        
        # Lightweight client-side filters server-applied for efficiency
        if agent:
            rows = [r for r in rows if (r.get("agent") or "") == agent]
        if section:
            rows = [r for r in rows if (r.get("section_name") or "") == section]
        
        from stephanie.tools.evidence_extractor import build_transfer_matrix_from_rows
        matrix = build_transfer_matrix_from_rows(rows)
        return matrix
        
    except Exception as e:
        _logger.error(f"Error building transfer matrix: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/api/run/{run_id}/timeline")
def api_run_learning_timeline(
    request: Request, 
    run_id: str,
    limit: int = Query(2000, ge=1, le=10000)
):
    """
    Get timeline of events for a run with learning context.
    """
    try:
        store = _store(request)
        rows = store.payloads_by_run(run_id, limit=limit)
        
        # Enrich with learning context
        out = []
        for r in rows:
            # Add learning context if available
            learning_context = {}
            if r.get("paper_id") and r.get("section_name"):
                learning_context["paper_context"] = f"{r['paper_id']} • {r['section_name']}"
            
            out.append({
                "ts": r.get("ts"),
                "event": r.get("event"),
                "paper_id": r.get("paper_id"),
                "section_name": r.get("section_name"),
                "agent": r.get("agent"),
                "subject": r.get("subject"),
                "learning_context": learning_context
            })
        
        return {"run_id": run_id, "events": out}
        
    except Exception as e:
        _logger.error(f"Error getting run timeline: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/api/learning_score")
def api_learning_score(
    request: Request,
    run_ids: str = Query("", description="Comma-separated run IDs")
):
    """
    Calculate learning scores across multiple runs for comparison.
    """
    try:
        store = _store(request)
        run_id_list = [rid for rid in run_ids.split(",") if rid.strip()]
        results = {}
        
        for run_id in run_id_list:
            rows = store.payloads_by_run(run_id, limit=2000)
            from stephanie.tools.evidence_extractor import build_transfer_matrix_from_rows
            matrix = build_transfer_matrix_from_rows(rows)
            results[run_id] = matrix["kpi"]["learning_score"]
        
        return results
        
    except Exception as e:
        _logger.error(f"Error calculating learning scores: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)