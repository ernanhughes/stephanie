from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse

import os
import yaml
from pathlib import Path

from stephanie.components.codecheck.engine import CodeCheckConfig, CodeCheckEngine
from stephanie.components.codecheck.agents.improver import (
    CodeCheckImproverAgent,
    CodeCheckImproverConfig,
)
import logging
log = logging.getLogger(__name__)


router = APIRouter()

IMPROVER_YAML_PATH = Path("config") / "agents" / "codecheck_improver.yaml"


def _load_improver_yaml() -> dict:
    """
    Brain-dead loader for config/agents/codecheck_improver.yaml.

    Returns the inner dict under 'codecheck_improver' or {} if missing.
    """
    if not IMPROVER_YAML_PATH.exists():
        return {}

    with IMPROVER_YAML_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # support both:
    #  - top-level `codecheck_improver: { ... }`
    #  - or just the bare config at root
    if "codecheck_improver" in raw:
        return raw["codecheck_improver"]
    return raw


# -----------------
# Helpers
# -----------------

def _safe_summary(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return summary or {}


def _safe_metrics_vector(file_obj) -> Dict[str, float]:
    """
    Extract metrics.vector safely from a CodeCheckFileORM or dict-like.
    """
    metrics = getattr(file_obj, "metrics", None)
    if not metrics:
        return {}
    vec = getattr(metrics, "vector", {}) or {}
    if isinstance(vec, dict):
        return vec
    return {}


# -----------------
# Runs
# -----------------


@router.get("/codecheck/runs", response_class=HTMLResponse)
def list_codecheck_runs(
    request: Request,
    status: Optional[str] = Query(default=None, description="Filter by run status"),
    branch: Optional[str] = Query(default=None, description="Filter by git branch"),
    limit: int = Query(default=50, ge=1, le=500),
):
    """
    List recent CodeCheck runs, similar to /casebooks. :contentReference[oaicite:3]{index=3}
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    # Expect MemoryTool to expose CodeCheckStore as memory.codecheck
    runs = memory.codecheck.list_runs(limit=limit)  # type: ignore[attr-defined]

    # Simple in-Python filtering for now
    filtered = []
    for r in runs:
        if status and getattr(r, "status", None) != status:
            continue
        if branch and getattr(r, "branch", None) != branch:
            continue
        filtered.append(r)

    items: List[Dict[str, Any]] = []
    for r in filtered:
        d = r.to_dict(include_files=False) if hasattr(r, "to_dict") else {}
        d.setdefault("status", getattr(r, "status", None))
        d.setdefault("branch", getattr(r, "branch", None))
        d.setdefault("commit_hash", getattr(r, "commit_hash", None))
        d.setdefault("created_ts", getattr(r, "created_ts", None))
        d["summary_metrics"] = _safe_summary(getattr(r, "summary_metrics", None))
        items.append(d)

    return templates.TemplateResponse(
        "/codecheck/runs_list.html",
        {
            "request": request,
            "runs": items,
            "filters": {
                "status": status,
                "branch": branch,
                "limit": limit,
            },
        },
    )


@router.get("/codecheck/runs/{run_id}", response_class=HTMLResponse)
def codecheck_run_detail(
    request: Request,
    run_id: str,
):
    """
    Show one CodeCheck run, summary metrics, and its files (with metrics & issue counts),
    similar in spirit to the casebook detail page. :contentReference[oaicite:4]{index=4}
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    run = memory.codecheck.get_run(run_id)  # type: ignore[attr-defined]
    if not run:
        return PlainTextResponse(f"CodeCheck run not found: {run_id}", status_code=404)

    run_d = run.to_dict(include_files=False) if hasattr(run, "to_dict") else {}
    run_d["summary_metrics"] = _safe_summary(getattr(run, "summary_metrics", None))

    # Files + metrics
    files = memory.codecheck.list_files_for_run(run_id=run_id, limit=5000)  # type: ignore[attr-defined]

    file_rows: List[Dict[str, Any]] = []
    metric_keys: set[str] = set()

    for f in files:
        fd = f.to_dict(include_metrics=True, include_issues=False) if hasattr(f, "to_dict") else {}
        vec = _safe_metrics_vector(f)
        fd["metrics_vector"] = vec
        for k in vec.keys():
            metric_keys.add(k)

        # Pre-compute an issue count per file for the table
        # (This uses list_issues_for_file; if that's not cheap, you can add a count column later.)
        issues = memory.codecheck.list_issues_for_file(f.id)  # type: ignore[attr-defined]
        fd["issue_count"] = len(issues) if issues is not None else 0

        file_rows.append(fd)

    # Sort metric keys for consistent column ordering
    metric_columns = sorted(metric_keys)

    # Issues at run level (for a sidebar, or bottom section)
    issues = memory.codecheck.list_issues_for_run(run_id=run_id, limit=2000)  # type: ignore[attr-defined]
    issue_rows: List[Dict[str, Any]] = []
    for iss in issues or []:
        if hasattr(iss, "to_dict"):
            issue_rows.append(iss.to_dict())
        else:
            issue_rows.append(
                {
                    "id": getattr(iss, "id", None),
                    "file_id": getattr(iss, "file_id", None),
                    "line": getattr(iss, "line", None),
                    "source": getattr(iss, "source", None),
                    "code": getattr(iss, "code", None),
                    "type": getattr(iss, "type", None),
                    "severity": getattr(iss, "severity", None),
                    "message": getattr(iss, "message", None),
                }
            )

    return templates.TemplateResponse(
        "/codecheck/run_detail.html",
        {
            "request": request,
            "run": run_d,
            "files": file_rows,
            "metric_columns": metric_columns,
            "issues": issue_rows,
        },
    )



@router.post("/codecheck/runs", response_class=RedirectResponse)
def start_codecheck_run(request: Request):
    """
    Kick off a new CodeCheck run and redirect to its detail page.

    For now we default repo_root to the current working directory.
    If you want to be explicit, change repo_root to your Stephanie root.
    """
    memory = request.app.state.memory
    container = request.app.state.container
    logger = request.app.state.logger


    # You can replace this with your actual project root if you prefer.
    repo_root = "./stephanie"

    cfg = CodeCheckConfig(repo_root=repo_root)
    engine = CodeCheckEngine(cfg=cfg, memory=memory, container=container, logger=logger)  # type: ignore[attr-defined]
    run_id = engine.run_analysis()

    # Redirect to the new run's detail page
    return RedirectResponse(url=f"/codecheck/runs/{run_id}", status_code=303)

@router.post("/codecheck/runs/{run_id}/improve", response_class=HTMLResponse)
async def improve_codecheck_run(request: Request, run_id: str):
    """
    Trigger the CodeCheckImproverAgent for a given run, using
    config/agents/codecheck_improver.yaml as the source of truth.
    """
    app = request.app
    memory = app.state.memory
    container = app.state.container
    logger = app.state.logger

    # 1) Load YAML config
    yaml_cfg = _load_improver_yaml()

    # 2) Map YAML → CodeCheckImproverConfig fields (brain-dead mapping)
    selection = yaml_cfg.get("selection") or {}
    model_cfg = yaml_cfg.get("model") or {}

    max_files = int(yaml_cfg.get("max_files", 10))
    max_suggestions_per_file = int(yaml_cfg.get("max_suggestions_per_file", 5))

    # recency_weight == YAML selection.recent_weight
    recency_weight = float(selection.get("recent_weight", 0.5))

    # Build simple weights from dirty_metrics list: each metric gets weight 1.0
    dirty_metrics = selection.get("dirty_metrics") or []
    if dirty_metrics:
        file_priority_weights = {name: 1.0 for name in dirty_metrics}
    else:
        # fallback to defaults from your dataclass
        file_priority_weights = {
            "security.bandit_high": 2.0,
            "readability.ruff_style": 1.0,
            "vibe.instruction_compliance": -1.0,
        }

    critic_model_name = model_cfg.get("model_name", "local-7b-code-critic")
    critic_temperature = float(model_cfg.get("temperature", 0.0))

    cfg = CodeCheckImproverConfig(
        run_id=run_id,
        max_files=max_files,
        max_suggestions_per_file=max_suggestions_per_file,
        file_priority_weights=file_priority_weights,
        recency_weight=recency_weight,
        critic_model_name=critic_model_name,
        critic_temperature=critic_temperature,
    )

    # 3) Run the agent
    agent = CodeCheckImproverAgent(cfg=cfg, memory=memory, container=container, logger=logger)

    # NOTE: your agent.run is currently async; call it with await.
    context = {
        "pipeline_run_id": run_id,
    }   
    result = await agent.run(context=context)

    log.info(
        "CodeCheckImproverAgent finished for run %s: %s",
        run_id,
        result,
    )

    # 4) Redirect back to the run detail page to see new suggestions
    return RedirectResponse(url=f"/codecheck/runs/{run_id}", status_code=303)
