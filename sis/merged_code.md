<!-- Merged Python Code Files -->


## File: __init__.py

`python
``n

## File: __main__.py

`python
import uvicorn


def main():
    uvicorn.run(
        "sis.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["sis"],
    )
``n

## File: config\config.yaml

`python
# sis/config/config.yaml

# Where SIS should pull insights from
source:
  type: "db"   # options: "db" | "logs"
  db_url: "postgresql://postgres:postgres@localhost:5432/stephanie"
  log_dir: "../logs"

# Database connection (for MemoryTool and ORM)
db:
  name: "co"
  user: "co"
  password: "co"
  host: "localhost"
  port: 5432

# Dashboard settings
dashboard:
  host: "0.0.0.0"
  port: 7860
  title: "📊 Stephanie Insight Dashboard"
  limit: 50   # how many recent insights to show

# API service (FastAPI/uvicorn)
api:
  host: "0.0.0.0"
  port: 8000
  reload: true

# Database tables (logical mapping for SIS views)
database:
  schema: "public"
  tables:
    insights: "insights"
    pipelines: "pipeline_runs"
    stages: "pipeline_stages"
    scores: "scores"
    cartridges: "cartridges"
    theorems: "theorems"

# Embeddings (to match MemoryTool usage)
embeddings:
  backend: "mxbai"   # options: mxbai | hnet | huggingface
  model: "mxbai-embed-large"
  dimension: 1024
  endpoint: "http://localhost:11434/api/embeddings"

# Logging
logging:
  level: "INFO"
  file: "sis.log"
``n

## File: dashboard.py

`python
# sis/dashboard.py

import gradio as gr
import sqlite3
import json

DB_PATH = "insight_data.db"

# Query recent insights
def fetch_insights():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, goal, agent, stage, notes, context, scores FROM insights ORDER BY timestamp DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return rows

# Render insights as a table
def view_insights():
    rows = fetch_insights()
    formatted = []
    for row in rows:
        timestamp, goal, agent, stage, notes, context_json, scores_json = row
        context = json.dumps(json.loads(context_json), indent=2)
        scores = json.dumps(json.loads(scores_json), indent=2)
        formatted.append([
            timestamp, goal, agent, stage, notes, context, scores
        ])
    return formatted

headers = ["Timestamp", "Goal", "Agent", "Stage", "Notes", "Context", "Scores"]

with gr.Blocks() as demo:
    gr.Markdown("# 📊 Stephanie Insight Dashboard")
    gr.Dataframe(headers=headers, value=view_insights, interactive=False)

if __name__ == "__main__":
    demo.launch()
``n

## File: db.py

`python
# sis/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from stephanie.models.base import Base

# You’ll load this from your config.yaml
def init_db(db_url: str):
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    return SessionLocal
``n

## File: main.py

`python
# sis/main.py
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stephanie.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from sis.routes import db, pipelines, models
import yaml

def load_config(path="sis/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
logger = JSONLogger("logs/sis.jsonl")
memory = MemoryTool(cfg=cfg, logger=logger)

app = FastAPI(title="Stephanie Insight System (SIS)")

# Put into app.state so routers can access without importing main
app.state.memory = memory
app.state.templates = Jinja2Templates(directory="sis/templates")
app.state.templates.env.globals["now"] = datetime.now

# Static assets
app.mount("/static", StaticFiles(directory="sis/static"), name="static")

# Include routers
app.include_router(pipelines.router)
app.include_router(models.router)
app.include_router(db.router)
``n

## File: routes\db.py

`python
from urllib import request
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import text
import logging
from io import StringIO
import csv 
from fastapi.responses import StreamingResponse

from sqlalchemy import text


router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/db-overview", response_class=HTMLResponse)
def db_overview(request: Request):
    memory = request.app.state.memory
    templates = request.app.state.templates
    session = memory.session

    # Get all tables in public schema
    tables = session.execute(text("""
        SELECT tablename 
        FROM pg_catalog.pg_tables 
        WHERE schemaname='public'
    """)).fetchall()

    table_data = []
    for (table_name,) in tables:
        try:
            count = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        except Exception:
            count = "N/A"
        table_data.append({"name": table_name, "count": count})

    return templates.TemplateResponse(
        "db_overview.html",
        {"request": request, "tables": table_data,  "active_page": "db"}
    )


@router.get("/db/table/{table_name}", response_class=HTMLResponse)
def table_detail(request: Request, table_name: str):
    memory = request.app.state.memory
    templates = request.app.state.templates

    try:
        # Use SQLAlchemy session directly
        session = memory.session

        # Run raw SQL (g What eneric, works for any table)
        result = session.execute(
            text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 100")
        )
        rows = result.fetchall()
        # Extract column names
        columns = result.keys() if rows else []

    except Exception as e:
        logger.error(f"Failed to load table {table_name}: {e}")
        return HTMLResponse(
            f"<h2>Error loading table '{table_name}': {e}</h2>", status_code=500
        )

    return templates.TemplateResponse(
        "table_detail.html",
        {
            "request": request,
            "table_name": table_name,
            "columns": columns,
            "rows": rows,
        },
    )

@router.get("/db/table/{table_name}/csv")
def table_detail_csv(request: Request, table_name: str):
    memory = request.app.state.memory
    try:
        session = memory.session
        result = session.execute(
            text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 100")
        )
        rows = result.mappings().all()
        columns = rows[0].keys() if rows else []

        # Write CSV
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[col] for col in columns])

        buffer.seek(0)
        filename = f"{table_name}_last100.csv"

        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        return HTMLResponse(f"<h2>Error exporting table '{table_name}': {e}</h2>", status_code=500)
``n

## File: routes\models.py

`python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sis.utils.model_tree import get_model_tree

router = APIRouter()

@router.get("/models", response_class=HTMLResponse)
def view_models(request: Request):
    templates = request.app.state.templates
    model_tree = "\n".join(get_model_tree("models"))
    return templates.TemplateResponse(
        "models.html",
        {"request": request, "model_tree": model_tree, "active_page": "models"}
    )
``n

## File: routes\pipelines.py

`python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import logging
from sis.utils.data_utils import get_run_config
from fastapi.responses import FileResponse, PlainTextResponse
import os
import tempfile


router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.get("/", response_class=HTMLResponse)
def list_pipelines(request: Request):
    memory = request.app.state.memory
    templates = request.app.state.templates

    logger.info("Fetching pipeline runs for listing")

    # Use MemoryTool accessor
    runs = memory.pipeline_runs.get_all(limit=50)

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
    prompts = memory.prompt.get_by_run_id(pipeline_id)
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
``n

## File: templates\base.html

`python
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}SIS{% endblock %}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container-fluid">
      <a class="navbar-brand" href="/">SIS</a>
      <div class="collapse navbar-collapse">
        <ul class="navbar-nav me-auto mb-2 mb-lg-0">
          <li class="nav-item">
            <a class="nav-link {% if active_page == 'pipelines' %}active{% endif %}" href="/">Pipelines</a>
          </li>
          <li class="nav-item">
            <a class="nav-link {% if active_page == 'models' %}active{% endif %}" href="/models">Models</a>
          </li>
          <li class="nav-item">
            <a class="nav-link {% if active_page == 'db' %}active{% endif %}" href="/db-overview">Database</a>
          </li>
        </ul>
      </div>
    </div>
  </nav>

  <div class="container mt-4">
    {% block content %}{% endblock %}
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
``n

## File: templates\context.html

`python
<div class="accordion-body">

  <!-- Metadata -->
  <p><strong>Strategy:</strong> {{ eval.strategy or "-" }}</p>
  <p><strong>Reasoning Strategy:</strong> {{ eval.reasoning_strategy or "-" }}</p>
  <p><strong>Target:</strong> {{ eval.target_type }} → {{ eval.target_id }}</p>

  <!-- Tabs -->
  <ul class="nav nav-tabs" id="evalTabs-{{ eval.id }}" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link active" id="scores-tab-{{ eval.id }}" data-bs-toggle="tab"
              data-bs-target="#scores-{{ eval.id }}" type="button" role="tab"
              aria-controls="scores-{{ eval.id }}" aria-selected="true">
        Scores
      </button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="attributes-tab-{{ eval.id }}" data-bs-toggle="tab"
              data-bs-target="#attributes-{{ eval.id }}" type="button" role="tab"
              aria-controls="attributes-{{ eval.id }}" aria-selected="false">
        Attributes
      </button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="context-tab-{{ eval.id }}" data-bs-toggle="tab"
              data-bs-target="#context-{{ eval.id }}" type="button" role="tab"
              aria-controls="context-{{ eval.id }}" aria-selected="false">
        Context
      </button>
    </li>
  </ul>

  <div class="tab-content mt-3" id="evalTabsContent-{{ eval.id }}">
    
    <!-- Scores Tab -->
    <div class="tab-pane fade show active" id="scores-{{ eval.id }}" role="tabpanel"
         aria-labelledby="scores-tab-{{ eval.id }}">
      {% if eval.dimension_scores %}
        <table class="table table-sm table-bordered">
          <thead>
            <tr><th>Dimension</th><th>Score</th><th>Source</th></tr>
          </thead>
          <tbody>
            {% for score in eval.dimension_scores %}
            <tr>
              <td>{{ score.dimension }}</td>
              <td>{{ "%.3f"|format(score.score) }}</td>
              <td>{{ score.source or "-" }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      {% elif eval.scores %}
        <pre class="bg-light p-2 rounded">{{ eval.scores | tojson(indent=2) }}</pre>
      {% else %}
        <p class="text-muted">No scores recorded.</p>
      {% endif %}
    </div>

    <!-- Attributes Tab -->
    <div class="tab-pane fade" id="attributes-{{ eval.id }}" role="tabpanel"
         aria-labelledby="attributes-tab-{{ eval.id }}">
      {% if eval.attributes %}
        <ul>
          {% for attr in eval.attributes %}
            <li><strong>{{ attr.key }}</strong>: {{ attr.value }}</li>
          {% endfor %}
        </ul>
      {% else %}
        <p class="text-muted">No attributes recorded.</p>
      {% endif %}
    </div>

    <!-- Context Tab -->
    <div class="tab-pane fade" id="context-{{ eval.id }}" role="tabpanel"
         aria-labelledby="context-tab-{{ eval.id }}">
      {% if eval.extra_data %}
        <pre class="bg-light p-2 rounded">{{ eval.extra_data | tojson(indent=2) }}</pre>
      {% else %}
        <p class="text-muted">No context data.</p>
      {% endif %}
    </div>

  </div>
</div>
``n

## File: templates\db_overview.html

`python
{% extends "base.html" %}

{% block content %}
<div class="container mt-4">
  <!-- Header -->
  <div class="d-flex justify-content-between align-items-center mb-4">
    <h1 class="mb-0">📊 Database Overview</h1>
    <a href="/" class="btn btn-secondary">⬅ Back to Pipelines</a>
  </div>

  {% if tables %}
    <!-- Stats summary -->
    <div class="alert alert-info">
      <strong>Total Tables:</strong> {{ tables|length }} |
      <strong>Total Rows:</strong> {{ tables | sum(attribute='count') }}
    </div>

    <!-- Table listing -->
    <table class="table table-striped table-bordered align-middle">
      <thead class="table-dark">
        <tr>
          <th style="width: 40%">Table</th>
          <th style="width: 20%">Row Count</th>
          <th style="width: 40%">Distribution</th>
        </tr>
      </thead>
      <tbody>
        {% set max_count = (tables | max(attribute='count')).count %}
        {% for table in tables %}
        <tr>
          <!-- Clickable table name -->
          <td>
            <a href="/db/table/{{ table.name }}" class="fw-bold text-decoration-none">
              📂 {{ table.name }}
            </a>
          </td>
          <!-- Row count badge -->
          <td>
            <span class="badge bg-primary">{{ table.count }}</span>
          </td>
          <!-- Relative progress bar -->
          <td>
            <div class="progress" style="height: 20px;">
              <div class="progress-bar bg-success" role="progressbar"
                style="width: {{ (table.count / max_count * 100) | round(1) }}%">
                {{ (table.count / max_count * 100) | round(1) }}%
              </div>
            </div>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <div class="alert alert-warning">⚠️ No tables found in database.</div>
  {% endif %}
</div>
{% endblock %}
``n

## File: templates\evaluation.html

`python
{% extends "base.html" %}

{% block content %}
<div class="container mt-4">
  <h1 class="mb-4">Evaluations for Pipeline Run {{ pipeline_run_id }}</h1>

  {% if evaluations %}
    <div class="accordion" id="evaluationAccordion">
      {% for eval in evaluations %}
      <div class="accordion-item">
        <h2 class="accordion-header" id="heading-{{ eval.id }}">
          <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-{{ eval.id }}" aria-expanded="false" aria-controls="collapse-{{ eval.id }}">
            <strong>{{ eval.agent_name }}</strong> ({{ eval.evaluator_name }}, {{ eval.model_name }})
            <span class="ms-3 text-muted small">{{ eval.created_at.strftime("%Y-%m-%d %H:%M") if eval.created_at else "-" }}</span>
          </button>
        </h2>
        <div id="collapse-{{ eval.id }}" class="accordion-collapse collapse" aria-labelledby="heading-{{ eval.id }}" data-bs-parent="#evaluationAccordion">
          <div class="accordion-body">
            <p><strong>Strategy:</strong> {{ eval.strategy or "-" }}</p>
            <p><strong>Model Name:</strong> {{ eval.model_name or "-" }}</p>
            <p><strong>Target:</strong> {{ eval.target_type }} → {{ eval.target_id }}</p>
            
            <h5>Scores</h5>
            {% if eval.dimension_scores %}
              <table class="table table-sm table-bordered">
                <thead>
                  <tr>
                    <th>Dimension</th>
                    <th>Score</th>
                    <th>Source</th>
                  </tr>
                </thead>
                <tbody>
                  {% for score in eval.dimension_scores %}
                  <tr>
                    <td>{{ score.dimension }}</td>
                    <td>{{ "%.3f"|format(score.score) }}</td>
                    <td>{{ score.source or "-" }}</td>
                  </tr>
                  {% endfor %}
                </tbody>
              </table>
            {% elif eval.scores %}
              <pre class="bg-light p-2 rounded">{{ eval.scores | tojson(indent=2) }}</pre>
            {% else %}
              <p class="text-muted">No scores recorded.</p>
            {% endif %}

            {% if eval.attributes %}
              <h6 class="mt-3">Attributes</h6>
              <ul>
                {% for attr in eval.attributes %}
                  <li><strong>{{ attr.key }}</strong>: {{ attr.value }}</li>
                {% endfor %}
              </ul>
            {% endif %}
          </div>
        </div>
      </div>
      {% endfor %}
    </div>
  {% else %}
    <div class="alert alert-warning">⚠️ No evaluations found for this pipeline run.</div>
  {% endif %}
</div>
{% endblock %}
``n

## File: templates\models.html

`python
{% extends "base.html" %}

{% block content %}
<div class="container mt-4">
  <div class="d-flex justify-content-between align-items-center mb-4">
    <h1 class="mb-0">📦 Installed Models</h1>
    <a href="/" class="btn btn-secondary">⬅ Back to Pipelines</a>
  </div>

  <pre class="bg-light p-3 rounded">{{ model_tree }}</pre>
</div>
{% endblock %}
``n

## File: templates\pipeline_detail.html

`python
{% extends "base.html" %}

{% block content %}
<div class="container mt-4">

  <!-- Header -->
  <div class="d-flex justify-content-between align-items-center mb-3">
    <h1 class="mb-0">Pipeline Run {{ run.id }}</h1>
    <a href="/" class="btn btn-outline-primary">⬅ Pipelines</a>
  </div>
  <p class="text-muted">Tag: {{ run.tag }} | Description: {{ run.description }}</p>

  <!-- Pipeline Stages -->
  <h3 class="mt-4">🔧 Pipeline Stages</h3>
  <div class="bg-light p-3 rounded">
    <pre class="mermaid">
flowchart LR
{% for stage in run.stages %}
  {{ "S" ~ stage.id }}[{{ stage.stage_name }} <br/> {{ stage.agent_class }}]
  {% if not loop.last %}
    {{ "S" ~ stage.id }} --> {{ "S" ~ run.stages[loop.index].id }}
  {% endif %}
{% endfor %}
    </pre>
  </div>

  <!-- Tabs -->
  <ul class="nav nav-tabs mt-4" id="pipelineTabs" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link active" id="report-tab" data-bs-toggle="tab" data-bs-target="#report" type="button" role="tab">📑 Report</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="prompts-tab" data-bs-toggle="tab" data-bs-target="#prompts" type="button" role="tab">💬 Prompts</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="evaluations-tab" data-bs-toggle="tab" data-bs-target="#evaluations" type="button" role="tab">📊 Evaluations</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="documents-tab" data-bs-toggle="tab" data-bs-target="#documents" type="button" role="tab">📚 Documents</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="config-tab" data-bs-toggle="tab" data-bs-target="#config" type="button" role="tab">⚙ Config</button>
    </li>
  </ul>

  <!-- Tab Contents -->
  <div class="tab-content mt-3" id="pipelineTabsContent">

    <!-- Report -->
    <div class="tab-pane fade show active" id="report" role="tabpanel">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <h4 class="mb-0">Pipeline Report</h4>
        {% if report %}
        <a href="/pipeline/{{ run.id }}/report/download" class="btn btn-sm btn-outline-primary">⬇ Download Full Report</a>
        {% endif %}
      </div>
      {% if report %}
      <div class="bg-light p-3 rounded" style="white-space: pre-wrap;">
        {{ report | safe }}
      </div>
      {% else %}
      <p class="text-muted">No report available for this run.</p>
      {% endif %}
    </div>

    <!-- Prompts -->
    <div class="tab-pane fade" id="prompts" role="tabpanel">
      {% if run.prompts %}
      <table class="table table-striped">
        <thead>
          <tr>
            <th>Agent</th>
            <th>Prompt</th>
            <th>Response</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {% for prompt in run.prompts %}
          <tr>
            <td>{{ prompt.agent_name }}</td>
            <td><code>{{ prompt.prompt_key }}</code></td>
            <td style="max-width: 600px; white-space: pre-wrap;">{{ prompt.response_text }}</td>
            <td>{{ prompt.timestamp.strftime("%Y-%m-%d %H:%M") if prompt.timestamp else "-" }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <p class="text-muted">No prompts recorded for this pipeline.</p>
      {% endif %}
    </div>

    <!-- Evaluations -->
    <div class="tab-pane fade" id="evaluations" role="tabpanel">
        <a class="btn btn-outline-primary" href="/pipeline/{{ run.id }}/scores">
        View Scores Grid
        </a>
      {% if run.evaluations %}

      <div class="accordion" id="evaluationAccordion">
        {% for eval in run.evaluations %}
        <div class="accordion-item">
          <h2 class="accordion-header" id="heading-{{ eval.id }}">
            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-{{ eval.id }}">
              {{ eval.evaluator_name }} ({{ eval.target_id }}, {{ eval.target_type }})
            </button>
          </h2>
          <div id="collapse-{{ eval.id }}" class="accordion-collapse collapse" data-bs-parent="#evaluationAccordion">
            <div class="accordion-body">
              <p><strong>Agent:</strong> {{ eval.agent_name or "-" }}</p>
              {% if eval.dimension_scores %}
              <table class="table table-sm table-bordered">
                <thead>
                  <tr><th>Dimension</th><th>Score</th><th>Source</th></tr>
                </thead>
                <tbody>
                  {% for score in eval.dimension_scores %}
                  <tr>
                    <td>{{ score.dimension }}</td>
                    <td>{{ "%.3f"|format(score.score) if score.score is not none else "-" }}</td>
                    <td>{{ score.source or "-" }}</td>
                  </tr>
                  {% endfor %}
                </tbody>
              </table>
              {% else %}
              <p class="text-muted">No scores recorded.</p>
              {% endif %}
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
      {% else %}
      <p class="text-muted">No evaluations recorded for this pipeline.</p>
      {% endif %}
    </div>

    <!-- Documents -->
    <div class="tab-pane fade" id="documents" role="tabpanel">
      {% if documents %}
      <div class="list-group">
        {% for key, doc in documents.items() %}
        <div class="list-group-item">
          <h6><span class="badge bg-secondary">{{ doc.target_type }}</span> ID: {{ doc.id }}</h6>
          <p class="mb-1 text-muted small">Target Key: {{ key }}</p>
          <pre class="bg-light p-2 rounded" style="max-height: 200px; overflow-y: auto;">{{ doc.text }}</pre>
        </div>
        {% endfor %}
      </div>
      {% else %}
      <div class="alert alert-warning">⚠️ No documents found for this pipeline run.</div>
      {% endif %}
    </div>

    <!-- Config -->
    <div class="tab-pane fade" id="config" role="tabpanel">
      {% if config_yaml %}
      <pre class="bg-dark text-light p-3 rounded" style="max-height: 500px; overflow-y: auto; white-space: pre-wrap;">{{ config_yaml }}</pre>
      {% else %}
      <p class="text-muted">No pipeline configuration available.</p>
      {% endif %}
    </div>

  </div> <!-- end tab-content -->

</div>

<!-- Mermaid Support -->
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true });
</script>
{% endblock %}
``n

## File: templates\pipeline_documents.html

`python
{% extends "base.html" %}

{% block content %}
<div class="container mt-4">
  <h1 class="mb-4">Documents for Pipeline Run {{ pipeline_run_id }}</h1>

  {% if documents %}
    <div class="list-group">
      {% for key, doc in documents.items() %}
      <div class="list-group-item">
        <h5 class="mb-1">
          <span class="badge bg-secondary">{{ doc.target_type }}</span>
          ID: {{ doc.id }}
        </h5>
        <p class="mb-1 text-muted small">
          Target Key: {{ key }}
        </p>
        <pre class="bg-light p-2 rounded" style="max-height: 200px; overflow-y: auto;">
{{ doc.text }}
        </pre>
      </div>
      {% endfor %}
    </div>
  {% else %}
    <div class="alert alert-warning">⚠️ No documents found for this pipeline run.</div>
  {% endif %}
</div>
{% endblock %}
``n

## File: templates\pipeline_scores.html

`python
{% extends "base.html" %}

{% block content %}
<div class="container mt-4">
  <h1>Scores Grid for Pipeline Run {{ pipeline_run_id }}</h1>

  {% if rows %}
  <div class="table-responsive mt-4">
    <table class="table table-bordered table-striped align-middle">
      <thead class="table-dark">
        <tr>
          <th>Evaluation ID</th>
          <th>Agent</th>
          <th>Evaluator</th>
          <th>Model</th>
          {% for dim in dimensions %}
          <th>{{ dim }}</th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for row in rows %}
        <tr>
          <td>{{ row.id }}</td>
          <td>{{ row.agent }}</td>
          <td>{{ row.evaluator }}</td>
          <td>{{ row.model }}</td>
          {% for dim in dimensions %}
          <td>
            {% if row.scores[dim] is not none %}
              {{ "%.3f"|format(row.scores[dim]) }}
            {% else %}
              <span class="text-muted">–</span>
            {% endif %}
          </td>
          {% endfor %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <div class="alert alert-warning">⚠️ No scores found for this pipeline run.</div>
  {% endif %}
</div>
{% endblock %}
``n

## File: templates\pipelines.html

`python
{# sis/templates/pipelines.html #}
{% extends "base.html" %}

{% block content %}
<div class="container mt-4">
  <div class="d-flex justify-content-between align-items-center mb-4">
    <h1 class="mb-0">📊 Pipeline Runs</h1>
    <a href="/models" class="btn btn-outline-primary">View Models</a>
  </div>

  {% if runs %}
    <table class="table table-striped table-hover align-middle">
      <thead class="table-dark">
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Tag</th>
          <th>Description</th>
          <th>Started</th>
          <th>Duration</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {% for run in runs %}
        <tr>
          <td>{{ run.id }}</td>
          <td>{{ run.name or "-" }}</td>
          <td>{{ run.tag or "-" }}</td>
          <td>{{ run.description or "-" }}</td>
          <td>{{ run.created_at.strftime("%Y-%m-%d %H:%M:%S") if run.created_at else "-" }}</td>
          <td>
            {% if run.stages %}
              {% set start = run.stages[0].timestamp %}
              {% set end = run.stages[-1].timestamp %}
              {% if start and end %}
                {{ (end - start).total_seconds() | int }}s
              {% else %}
                -
              {% endif %}
            {% else %}
              -
            {% endif %}
          </td>
          <td>
            <a href="/pipeline/{{ run.id }}" class="btn btn-sm btn-primary">View</a>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <div class="alert alert-warning">
      ⚠️ No pipelines found.
    </div>
  {% endif %}
</div>
{% endblock %}
``n

## File: templates\prompts.html

`python
{% extends "base.html" %}

{% block content %}
<div class="container mt-4">
  <h1 class="mb-4">Prompts & Responses</h1>

  {% if prompts %}
    <table class="table table-hover align-middle">
      <thead class="table-dark">
        <tr>
          <th style="width: 15%;">Agent</th>
          <th style="width: 20%;">Prompt Key</th>
          <th style="width: 55%;">Response</th>
          <th style="width: 10%;">Timestamp</th>
        </tr>
      </thead>
      <tbody>
        {% for prompt in prompts %}
        <tr>
          <td>{{ prompt.agent_name or "-" }}</td>
          <td>{{ prompt.prompt_key }}</td>
          <td style="white-space: pre-wrap; max-width: 800px;">
            <div class="mb-2">
              <small class="text-muted"><strong>Prompt:</strong> {{ prompt.prompt_text }}</small>
            </div>
            <div>
              <strong>Response:</strong>
              <div class="p-2 border rounded bg-light mt-1">
                {{ prompt.response_text or "-" }}
              </div>
            </div>
          </td>
          <td>{{ prompt.timestamp.strftime("%Y-%m-%d %H:%M") if prompt.timestamp else "-" }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <div class="alert alert-warning">⚠️ No prompts found.</div>
  {% endif %}
</div>
{% endblock %}
``n

## File: templates\table_detail.html

`python
{% extends "base.html" %}

{% block content %}
<div class="container mt-4">
  <!-- Header -->
  <div class="d-flex justify-content-between align-items-center mb-4">
    <h1 class="mb-0">📂 Table: {{ table_name }}</h1>
    <a href="/db-overview" class="btn btn-secondary">⬅ Back to Overview</a>
    <a href="/db/table/{{ table_name }}/csv" class="btn btn-success">⬇ Download CSV</a>
  </div>

  {% if rows %}
    <table class="table table-hover table-bordered table-sm">
      <thead class="table-dark">
        <tr>
          {% for col in columns %}
            <th>{{ col }}</th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for row in rows %}
        <tr>
          {% for col in columns %}
            <td style="max-width: 250px; white-space: pre-wrap; word-break: break-word;">
              {{ row[col] }}
            </td>
          {% endfor %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
    <p class="text-muted">Showing last {{ rows|length }} rows (ordered by <code>id</code> DESC).</p>
  {% else %}
    <div class="alert alert-warning">⚠️ No rows found for this table.</div>
  {% endif %}
</div>
{% endblock %}
``n

## File: utils\data_utils.py

`python
import copy
import json
import yaml


SENSITIVE_KEYS = {"password", "db_password", "secret", "api_key", "token"}

def sanitize_config(config: dict) -> dict:
    """
    Recursively walks the config dict and replaces sensitive values with '***'.
    """
    safe = copy.deepcopy(config)
    for k, v in safe.items():
        if isinstance(v, dict):
            safe[k] = sanitize_config(v)
        elif isinstance(v, list):
            safe[k] = [sanitize_config(i) if isinstance(i, dict) else i for i in v]
        else:
            if k.lower() in SENSITIVE_KEYS:
                safe[k] = "***"
    return safe

def get_run_config(run) -> dict:
    """
    Extracts and sanitizes the run configuration from the pipeline run dict.
    """
    config_yaml = None
    if run.run_config:
        try:
            if isinstance(run.run_config, str):
                config_dict = json.loads(run.run_config)
            else:
                config_dict = run.run_config
            safe_config = sanitize_config(config_dict)
            config_yaml = yaml.dump(safe_config, sort_keys=False, indent=2)
        except Exception as e:
            config_yaml = f"# Error converting config to YAML: {e}"

    return config_yaml
``n

## File: utils\model_tree.py

`python
# sis/utils/model_tree.py
import os

MODEL_TYPE_ICONS = {
    "svm": "🌀",
    "mrq": "🧠",
    "ebt": "🪜"
}

def get_icon(name, is_dir):
    name = name.lower()
    if is_dir:
        return "📁"
    elif "encoder.pt" in name:
        return "🧠"
    elif "model.pt" in name:
        return "🤖"
    elif "tuner.json" in name:
        return "🎚️"
    elif "_scaler.joblib" in name:
        return "📏"
    elif name.endswith("meta.json"):
        return "⚙️"
    elif name.endswith(".log"):
        return "📋"
    elif name.endswith((".yaml", ".yml")):
        return "📄"
    elif name.endswith(".md"):
        return "📘"
    elif name.endswith(".json"):
        return "🗂️"
    elif name.endswith(".pt"):
        return "📦"
    elif name.endswith(".joblib") or name.endswith(".pkl"):
        return "📦"
    elif name.endswith(".onnx"):
        return "📐"
    elif name.endswith(".txt"):
        return "📝"
    elif name.endswith(".py"):
        return "🐍"
    else:
        return "📦"

def get_model_type_icon(name):
    return MODEL_TYPE_ICONS.get(name.lower(), "📦")

def build_tree(root_path, indent="", is_top_level=True):
    """Return a string tree for templates instead of printing."""
    lines = []
    try:
        entries = sorted(os.listdir(root_path))
    except PermissionError:
        lines.append(indent + "└── 🔒 Permission Denied")
        return lines

    for i, entry in enumerate(entries):
        path = os.path.join(root_path, entry)
        is_dir = os.path.isdir(path)
        connector = "└──" if i == len(entries) - 1 else "├──"

        if is_top_level and is_dir:
            icon = get_model_type_icon(entry)
        else:
            icon = get_icon(entry, is_dir)

        lines.append(f"{indent}{connector} {icon}  {entry}")

        if is_dir:
            extension = "    " if i == len(entries) - 1 else "│   "
            lines.extend(build_tree(path, indent + extension, is_top_level=False))
    return lines

def get_model_tree(base_dir="models"):
    if not os.path.exists(base_dir):
        return [f"❌ Directory '{base_dir}' does not exist."]
    return [f"📦 {base_dir}"] + build_tree(base_dir)
``n
