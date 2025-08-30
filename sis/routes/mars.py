from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

router = APIRouter()

@router.get("/mars", response_class=HTMLResponse)
def mars_results(request: Request):
    memory = request.app.state.memory
    templates = request.app.state.templates

    results = memory.mars_results.get_recent(100)

    return templates.TemplateResponse(
        "/mars/list.html",
        {
            "request": request,
            "results": [r.to_dict() for r in results]
        }
    )

@router.get("/mars/{pipeline_run_id}", response_class=HTMLResponse)
def mars_result_detail(request: Request, pipeline_run_id: int):
    """
    Show detailed results for a single MARSResult entry,
    including underlying scorer values per scorable.
    """
    memory = request.app.state.memory
    templates = request.app.state.templates
    
    mars_result = memory.mars_results.get_by_run_id(pipeline_run_id)
    print(f"found {len(mars_result)} MARS results for pipeline_run_id {pipeline_run_id}")
    if not mars_result:
        return PlainTextResponse("MARS result not found", status_code=404)

    evaluations = memory.evaluations.get_by_run_id(pipeline_run_id)

    # Flatten out scores
    rows = []
    all_dimensions = set()
    ids = [eval["id"] for eval in evaluations]
    scores = memory.scores.get_by_evaluation_ids(ids)
    for score in scores:
        all_dimensions.add(score.dimension)
    dimensions = sorted(all_dimensions)

    rows = []
    for e in evaluations:
        row = {
            "id": e["id"] if isinstance(e, dict) else e.id,
            "agent": e["agent_name"] if isinstance(e, dict) else e.agent_name,
            "evaluator": e["evaluator_name"]
            if isinstance(e, dict)
            else e.evaluator_name,
            "model": e["model_name"] if isinstance(e, dict) else e.model_name,
            "scores": dict.fromkeys(dimensions),
        }

        # Attach scores for this eval
        related_scores = [s for s in scores if s.evaluation_id == row["id"]]
        for s in related_scores:
            row["scores"][s.dimension] = s.score

        rows.append(row)
    score_bundles = rows

    return templates.TemplateResponse(
        "/mars/detail.html",
        {
            "request": request,
            "mars": mars_result,
            "score_bundles": score_bundles,
            "rows": rows,
            "dimensions": dimensions,
        },
    )
