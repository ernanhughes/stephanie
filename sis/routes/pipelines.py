from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import logging
from sis.utils.data_utils import get_run_config
from fastapi.responses import FileResponse, PlainTextResponse
import os
import tempfile
import yaml

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.get("/", response_class=HTMLResponse)
def list_pipelines(request: Request):
    memory = request.app.state.memory
    templates = request.app.state.templates

    logger.info("Fetching pipeline runs for listing")

    # Use MemoryTool accessor
    runs = memory.pipeline_runs.list_runs_with_stages(limit=50)

    return templates.TemplateResponse(
        "pipelines.html",
        {"request": request, "runs": runs, "active_page": "pipelines"},
    )


@router.get("/pipeline/{pipeline_id}", response_class=HTMLResponse)
def pipeline_detail(request: Request, pipeline_id: int):
    memory = request.app.state.memory
    templates = request.app.state.templates

    logger.info(f"Fetching details for pipeline run {pipeline_id}")
    # Load the pipeline run
    run = memory.pipeline_runs.get_by_run_id(pipeline_id)
    if not run:
        return HTMLResponse("<h2>Pipeline not found</h2>", status_code=404)

    logger.info(f"Loaded pipeline run: {run}")
    # Related objects via MemoryTool
    prompts = memory.prompts.get_by_run_id(pipeline_id)
    evaluations = memory.evaluations.get_by_run_id(pipeline_id)
    ids = [eval["id"] for eval in evaluations]
    scores = memory.scores.get_by_evaluation_ids(ids)
    report = memory.reports.get_content(run.id)
    report_path = memory.reports.get_path(run.id)

    for e in evaluations:
        e["scores"] = [
            s.to_dict() for s in scores if s.evaluation_id == e["id"]
        ]

    stages = memory.pipeline_stages.get_by_run_id(pipeline_id)
    documents = memory.pipeline_references.get_documents_by_run_id(
        pipeline_id, memory, limit=100
    )
    config_yaml = get_run_config(run)

    cartridges = memory.cartridges.get_run_id(pipeline_id)
    theorems = memory.theorems.get_by_run_id(pipeline_id)
    print(f"Found {len(theorems)} theorems for pipeline run {pipeline_id}")

    # Expand with triples
    cartridges = [
        {
            **c.to_dict(),
            "triples": [t.to_dict() for t in c.triples_rel] if c.triples_rel else []
        }
        for c in cartridges
    ]
    print(f"Found {len(cartridges)} cartridges for pipeline run {pipeline_id}")

    return templates.TemplateResponse(
        "pipeline_detail.html",
        {
            "request": request,
            "run": run,
            "prompts": prompts,
            "evaluations": evaluations,
            "stages": stages,
            "documents": documents,
            "report": report,
            "report_path": report_path,
            "cartridges": cartridges,
            "theorems": theorems,
            "config_yaml": config_yaml,
        },
    )

@router.get("/pipeline/{pipeline_id}/evaluations", response_class=HTMLResponse)
def pipeline_evaluations(request: Request, pipeline_id: int):
    memory = request.app.state.memory
    templates = request.app.state.templates

    evaluations = memory.evaluations.get_by_run_id(pipeline_id)

    print(
        f"Found {len(evaluations)} evaluations for pipeline run {pipeline_id}"
    )
    logger.info(
        f"Found {len(evaluations)} evaluations for pipeline run {pipeline_id}"
    )
    # Attach dimension scores for each evaluation
    for eval in evaluations:
        eval.dimension_scores = memory.scores.get_scores_for_evaluation(
            eval.id
        )

    return templates.TemplateResponse(
        "evaluations.html",
        {
            "request": request,
            "pipeline_run_id": pipeline_id,
            "evaluations": evaluations,
        },
    )


@router.get("/pipeline/{pipeline_id}/scores", response_class=HTMLResponse)
def pipeline_scores(request: Request, pipeline_id: int):
    memory = request.app.state.memory
    templates = request.app.state.templates

    # Fetch evaluations + scores
    evaluations = memory.evaluations.get_by_run_id(pipeline_id)

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
            "scores": {dim: None for dim in dimensions},
        }

        # Attach scores for this eval
        related_scores = [s for s in scores if s.evaluation_id == row["id"]]
        for s in related_scores:
            row["scores"][s.dimension] = s.score

        rows.append(row)

    return templates.TemplateResponse(
        "pipeline_scores.html",
        {
            "request": request,
            "pipeline_run_id": pipeline_id,
            "dimensions": dimensions,
            "rows": rows,
        },
    )


@router.get("/pipeline/{pipeline_id}/documents", response_class=HTMLResponse)
def pipeline_documents(request: Request, pipeline_id: int):
    memory = request.app.state.memory
    templates = request.app.state.templates

    logger.info(f"Fetching documents for pipeline run {pipeline_id}")
    # Fetch the top 100 referenced documents for this pipeline run
    documents = memory.pipeline_references.get_documents_by_run_id(
        pipeline_id, memory, limit=100
    )

    logger.info(
        f"Found {len(documents)} documents for pipeline run {pipeline_id}"
    )

    return templates.TemplateResponse(
        "pipeline_documents.html",
        {
            "request": request,
            "pipeline_run_id": pipeline_id,
            "documents": documents,
        },
    )


@router.get("/pipeline/{pipeline_id}/report/download")
def download_report(request: Request, pipeline_id: int):
    memory = request.app.state.memory

    report = memory.reports.get_content(pipeline_id)
    if not report:
        return PlainTextResponse("Report not found", status_code=404)

    # Save to a temporary file
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"pipeline_{pipeline_id}_report.md"
    )
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(report or "")

    return FileResponse(
        path=tmp_path,
        filename=f"pipeline_{pipeline_id}_report.md",
        media_type="text/markdown",
    )


@router.get("/pipeline/{pipeline_id}/cartridges", response_class=HTMLResponse)
def pipeline_cartridges(request: Request, pipeline_id: int):
    memory = request.app.state.memory
    templates = request.app.state.templates

    cartridges = memory.cartridges.get_run_id(pipeline_id)

    # Expand with triples
    cartridges = [
        {
            **c.to_dict(),
            "triples": [t.to_dict() for t in c.triples_rel] if c.triples_rel else []
        }
        for c in cartridges
    ]
    print(f"Found {len(cartridges)} cartridges for pipeline run {pipeline_id}")

    return templates.TemplateResponse(
        "pipeline_cartridges.html",
        {
            "request": request,
            "pipeline_run_id": pipeline_id,
            "cartridges": cartridges,
        },
    )


@router.get("/pipeline/{pipeline_id}/theorems", response_class=HTMLResponse)
def pipeline_theorems(request: Request, pipeline_id: int):
    memory = request.app.state.memory
    templates = request.app.state.templates

    theorems = memory.theorems.get_by_run_id(pipeline_id)

    return templates.TemplateResponse(
        "pipeline_theorems.html",
        {
            "request": request,
            "pipeline_run_id": pipeline_id,
            "theorems": theorems,
        },
    )
