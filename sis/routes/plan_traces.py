from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()

@router.get("/plan_traces", response_class=HTMLResponse)
def list_plan_traces(request: Request):
    templates = request.app.state.templates
    memory = request.app.state.memory
    
    # Fetch all plan traces (maybe limit to recent N)
    traces = memory.plan_traces.get_all(limit=50)  # implement in your store
    
    return templates.TemplateResponse(
        "/plan_traces/plan_traces.html",
        {"request": request, "traces": traces, "active_page": "plan_traces"}
    )

@router.get("/plan_traces/{run_id}", response_class=HTMLResponse)
def view_plan_trace(request: Request, run_id: str):
    templates = request.app.state.templates
    memory = request.app.state.memory

    print("Fetching PlanTrace for run_id:", run_id)
    
    trace = memory.plan_traces.get_by_run_id(run_id)
    links = memory.plan_traces.get_reuse_links_for_trace(trace.trace_id)
    links = [(link.parent_trace_id, link.child_trace_id, link.created_at) for link in links]
    goal_text = memory.plan_traces.get_goal_text(run_id)
    if not trace:
        return HTMLResponse(f"<h3>❌ PlanTrace not found: {run_id}</h3>", status_code=404)
    
    stages = memory.pipeline_stages.get_by_run_id(run_id)
    

    # Execution steps are embedded in trace.execution_steps
    steps = trace.execution_steps or []
    
    return templates.TemplateResponse(
        "/plan_traces/plan_trace.html",
        {
            "request": request,
            "trace": trace,
            "steps": steps,
            "stages": stages,
            "links": links,
            "goal_text": goal_text,
            "active_page": "plan_traces"
        }
    )


@router.post("/plan_traces/{trace_id}/revisions")
async def add_revision(trace_id: str, request: Request):
    form = await request.form()
    revision_type = form.get("revision_type", "feedback")
    revision_text = form.get("revision_text", "")
    memory = request.app.state.memory
    memory.plan_traces.add_revision(trace_id, revision_type, revision_text, "user")
    return RedirectResponse(f"/plan_traces/{trace_id}", status_code=303)
