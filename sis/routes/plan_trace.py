# sis/routes/plan_traces.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
import logging

from stephanie.data.plan_trace import PlanTrace

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_class=HTMLResponse)
def list_plan_traces(request: Request, similar_to: str = None):
    """
    List all plan traces, or find traces similar to another one.
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    try:
        if similar_to:
            base_trace = memory.plan_traces.get_by_trace_id(similar_to)
            if not base_trace:
                return templates.TemplateResponse(
                    "plan_traces.html",
                    {"request": request, "traces": [], "message": f"Trace {similar_to} not found"},
                )

            # Use embedding similarity
            query_text = (base_trace.final_output_text or "") + " " + (base_trace.plan_signature or "")
            traces = memory.plan_traces.get_similar_traces(query_text, top_k=20, embedding=memory.embedding)
            message = f"Traces similar to {similar_to}"
        else:
            traces = memory.plan_traces.get_all(limit=100)
            message = None

        return templates.TemplateResponse(
            "plan_traces.html",
            {"request": request, "traces": traces, "message": message},
        )

    except Exception as e:
        logger.exception("Error listing plan traces")
        return templates.TemplateResponse(
            "plan_traces.html",
            {"request": request, "traces": [], "message": f"Error: {str(e)}"},
        )


@router.get("/{trace_id}", response_class=HTMLResponse)
def view_plan_trace(request: Request, trace_id: str):
    """
    View details of a specific plan trace, including pipeline stages and execution steps.
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    try:
        trace: PlanTrace = memory.plan_traces.get_by_trace_id(trace_id)
        if not trace:
            return templates.TemplateResponse(
                "plan_trace.html",
                {"request": request, "trace": None, "stages": [], "steps": [], "error": "Trace not found"},
            )

        # Collect related stages + steps
        stages =  memory.pipeline_stages.get_by_run_id(trace.pipeline_run_id) or []
        steps = trace.execution_steps or []

        return templates.TemplateResponse(
            "/plan_traces/plan_trace.html",
            {
                "request": request,
                "trace": trace,
                "stages": stages,
                "steps": steps,
            },
        )

    except Exception as e:
        logger.exception("Error viewing plan trace")
        return templates.TemplateResponse(
            "plan_trace.html",
            {"request": request, "trace": None, "stages": [], "steps": [], "error": str(e)},
        )
