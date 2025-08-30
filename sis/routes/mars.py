# sis/routes/mars.py (or wherever your router lives)
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

router = APIRouter()

@router.get("/mars", response_class=HTMLResponse)
def mars_results(request: Request):
    memory = request.app.state.memory
    templates = request.app.state.templates
    results = memory.mars_results.get_recent(100)
    return templates.TemplateResponse(
        "/mars/list.html",
        {"request": request, "results": [r.to_dict() if hasattr(r, "to_dict") else r for r in results]},
    )

@router.get("/mars/{pipeline_run_id}", response_class=HTMLResponse)
def mars_result_detail(
    request: Request,
    pipeline_run_id: int,
    dimension: str | None = Query(default=None, description="Optional dimension focus"),
):
    """
    Show detailed results for a single pipeline run:
    - One or more per-dimension MARS result entries
    - Underlying evaluation rows and their per-dimension scores
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    raw = memory.mars_results.get_by_run_id(pipeline_run_id)
    if not raw:
        return PlainTextResponse("MARS result not found", status_code=404)

    # Normalize to list[dict] and extract the stored per-dimension `result`
    mars_list = []
    for r in raw:
        d = r.to_dict() if hasattr(r, "to_dict") else r
        # many stores keep {"result": {...}}; fallback to d if already flat
        res = d.get("result") or d
        # require minimal expected fields
        if isinstance(res, dict) and res.get("dimension"):
            mars_list.append(res)

    # Optional filter by dimension via query param
    highlight_dim = dimension
    if highlight_dim:
        mars_list = [m for m in mars_list if m.get("dimension") == highlight_dim] or mars_list

    # Build underlying evaluation table
    evaluations = memory.evaluations.get_by_run_id(pipeline_run_id)
    if not evaluations:
        evaluations = []

    # Collect all score rows (EvaluationORM → row dict) and dimensions present
    rows = []
    all_dimensions = set()

    eval_ids = [e["id"] if isinstance(e, dict) else e.id for e in evaluations]
    score_rows = memory.scores.get_by_evaluation_ids(eval_ids) if eval_ids else []
    for s in score_rows:
        all_dimensions.add(s.dimension)
    dimensions = sorted(all_dimensions)

    # Build rows with a dense score dict per evaluation
    for e in evaluations:
        row = {
            "id": e["id"] if isinstance(e, dict) else e.id,
            "agent": e["agent_name"] if isinstance(e, dict) else e.agent_name,
            "evaluator": e["evaluator_name"] if isinstance(e, dict) else e.evaluator_name,
            "model": e["model_name"] if isinstance(e, dict) else e.model_name,
            "scores": dict.fromkeys(dimensions),
        }
        related = [s for s in score_rows if s.evaluation_id == row["id"]]
        for s in related:
            print(s.to_dict() if isinstance(s, dict) else s)
            row["scores"][s.dimension] = s.get("score") if isinstance(s, dict) else s.score
        rows.append(row)

    # Simple summary across dimensions shown
    if mars_list:
        avg_agreement = sum(float(m.get("agreement_score", 0.0)) for m in mars_list) / max(len(mars_list), 1)
        avg_std = sum(float(m.get("std_dev", 0.0)) for m in mars_list) / max(len(mars_list), 1)
    else:
        avg_agreement = 0.0
        avg_std = 0.0

    return templates.TemplateResponse(
        "/mars/detail.html",
        {
            "request": request,
            "pipeline_run_id": pipeline_run_id,
            "mars_list": mars_list,          # <-- iterate this in template
            "rows": rows,
            "dimensions": dimensions,
            "highlight_dim": highlight_dim,  # optional UI highlight
            "summary": {"avg_agreement": avg_agreement, "avg_std": avg_std},
        },
    )
