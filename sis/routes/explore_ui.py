# sis/routes/explore_ui.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse

from stephanie.components.gap.risk.explore.runner import ExplorationRunner

router = APIRouter(prefix="/explore", tags=["explore-ui"])

def _default_seeds() -> List[str]:
    return [
        "What are subtle failure modes in retrieval-augmented generation?",
        "How do we detect emergent out-of-domain drift in chat agents?",
        "Ways to measure faithfulness without external ground truth?",
        "How can small monitors track disagreement with a large model?",
        "Design a visual badge encoding risk/uncertainty at a glance.",
    ]

def _parse_seeds(raw: str, limit: int) -> List[str]:
    lines = [ln.strip() for ln in (raw or "").splitlines()]
    lines = [ln for ln in lines if ln]
    # dedupe preserving order
    seen, out = set(), []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
        if len(out) >= limit:
            break
    return out

def _interest(novelty: float, reasons: Dict[str, float]) -> float:
    # Simple interest score: novelty + portions of delta/ood risk
    rd = float(reasons.get("risk_delta", 0.0))
    ro = float(reasons.get("risk_ood", 0.0))
    return 0.5 * float(novelty) + 0.3 * rd + 0.2 * ro

def _runner(request: Request, profile: Optional[str]) -> ExplorationRunner:
    key = f"_explore_runner::{profile or 'chat.standard'}"
    runner = getattr(request.app.state, key, None)
    if runner is None:
        runner = ExplorationRunner(request.app.state.container, profile=profile or "chat.standard")
        setattr(request.app.state, key, runner)
    return runner

@router.get("", response_class=HTMLResponse)
async def form_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse("/explore/form.html", {"request": request})

@router.post("/run", response_class=HTMLResponse)
async def run_explore(
    request: Request,
    seeds: str = Form(""),
    sample_n: int = Form(5),
    k_triggers: int = Form(4),
    divergence: float = Form(0.8),
    policy_profile: Optional[str] = Form(None),
    model_alias: str = Form("chat-hrm"),      # (for provenance only; runner uses container.chat_sampler)
    monitor_alias: str = Form("tiny-monitor") # used in orchestrator evaluate()
):
    templates = request.app.state.templates
    seed_list = _parse_seeds(seeds, limit=max(1, int(sample_n)))
    if not seed_list:
        seed_list = _default_seeds()[: max(1, int(sample_n))]

    runner = _runner(request, policy_profile)
    results: List[Dict[str, Any]] = []
    # NOTE: synchronous loop (keeps Option A simple). Start small (≤5) to avoid long requests.
    for seed in seed_list:
        rows = await runner.explore_goal(seed, k_triggers=int(k_triggers), divergence=float(divergence))
        # compute interest per row and attach convenience fields
        for rec in rows:
            rec["interest"] = _interest(rec.get("novelty", 0.0), rec.get("reasons", {}))
            rec["goal_seed"] = seed
            rec["prompt_used"] = rec.get("prompt_used", rec.get("seed_goal", seed))
        results.extend(rows)

    # sort by interest desc; cap to 100 rows for sanity
    results.sort(key=lambda r: r.get("interest", 0.0), reverse=True)
    results = results[:100]

    ctx = {
        "request": request,
        "results": results,
        "k_triggers": k_triggers,
        "divergence": divergence,
        "policy_profile": policy_profile,
        "sample_n": sample_n,
    }
    return templates.TemplateResponse("/explore/result.html", ctx)
