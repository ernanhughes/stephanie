from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/plan_traces", response_class=HTMLResponse)
def list_plan_traces(request: Request):
    templates = request.app.state.templates
    memory = request.app.state.memory
    
    # Fetch all plan traces (maybe limit to recent N)
    traces = memory.plan_traces.get_all(limit=50)  # implement in your store
    
    return templates.TemplateResponse(
        "plan_traces.html",
        {"request": request, "traces": traces, "active_page": "plan_traces"}
    )

@router.get("/plan_traces/{run_id}", response_class=HTMLResponse)
def view_plan_trace(request: Request, run_id: str):
    templates = request.app.state.templates
    memory = request.app.state.memory

    print("Fetching PlanTrace for run_id:", run_id)
    
    trace = memory.plan_traces.get_by_run_id(run_id)
    if not trace:
        return HTMLResponse(f"<h3>❌ PlanTrace not found: {run_id}</h3>", status_code=404)
    
    # Execution steps are embedded in trace.execution_steps
    steps = trace.execution_steps or []
    
    return templates.TemplateResponse(
        "plan_trace_detail.html",
        {
            "request": request,
            "trace": trace,
            "steps": steps,
            "active_page": "plan_traces"
        }
    )
