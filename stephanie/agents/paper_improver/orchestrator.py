# stephanie/agents/paper_improver/orchestrator.py
# Orchestrator — End-to-end spec+plan → code+text → goals → (optional) PRs with gating
from __future__ import annotations

import argparse
import json
import platform
import random
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from stephanie.agents.knowledge.improver import Improver
from stephanie.knowledge.casebook_store import CaseBookStore
# stephanie/agents/paper_improver/orchestrator.py (diffs only)
from stephanie.knowledge.knowledge_bus import KnowledgeBus

from ...zeromodel.vpm_controller import VPMController
from .code_improver import CodeImprover
from .faithfulness import \
    FaithfulnessBot  # optional use; safe if not provided at CLI
from .goals import GoalScorer, build_templates_from_yaml, load_yaml
from .repo_link import RepoLink

# --------------------------- helpers ---------------------------

def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds") + "Z"

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def _append_ndjson(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _env_fingerprint() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

def _seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

# --------------------------- orchestrator ---------------------------

def run_paper_section(
    spec_path: str,
    plan_path: str,
    kb: KnowledgeBus = KnowledgeBus(),
    workdir: str = "./runs",
    backend: str = "torch",
    create_pr: bool = False,
    repo_root: str = "../..",
    goals_yaml: Optional[str] = None,
    goal_key: Tuple[str, str] = ("text", "academic_summary"),  # (kind, name)
    gate_pr_on_goal: bool = True,
    gate_min_score: float = 0.75,
    gate_allow_unmet: int = 0,
    paper_txt_path: Optional[str] = None,   # if provided, run faithfulness check
    faithfulness_topk: int = 5,
    skip_code: bool = False,
    skip_text: bool = False,
    dry_run_pr: bool = False,
    seed: int = 0,
) -> dict:
    """
    Run code + text improvers with telemetry, goals, (optional) faithfulness, and PR gating.

    Returns combined report dict and writes:
      - {workdir}/report.json
      - {workdir}/trace.ndjson
    """


    kb = KnowledgeBus()
    store = CaseBookStore()
    gs = GoalScorer()

    ci = CodeImprover(backend=backend, workdir=f"{workdir}/code", kb=kb, casebooks=store)
    ti = Improver(workdir=f"{workdir}/text", kb=kb, casebooks=store)

    # controller with goal scorer + persistence
    goal_scorer = lambda kind, name, dims: gs.score(kind, name, dims)
    vc = VPMController(..., goal_scorer=goal_scorer, state_path=f"{workdir}/vpm_state.json")

    t0 = time.time()
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    workdir_p = Path(workdir)
    trace_path = workdir_p / "trace.ndjson"

    _seed_everything(seed)

    # Load inputs
    spec = json.loads(Path(spec_path).read_text())
    plan = json.loads(Path(plan_path).read_text())

    # Goals/templates
    templates_cfg = load_yaml(goals_yaml) if goals_yaml else {}
    templates = build_templates_from_yaml(templates_cfg) if templates_cfg else None
    gs = GoalScorer(templates=templates)

    # Controllers & improvers
    vc = VPMController()
    ci = None if skip_code else CodeImprover(backend=backend, workdir=str(workdir_p / "code"))
    ti = None if skip_text else Improver(workdir=str(workdir_p / "text"))

    # Collect report
    report: Dict[str, Any] = {
        "run_id": run_id,
        "started": _ts(),
        "env": _env_fingerprint(),
        "inputs": {"spec": spec_path, "plan": plan_path, "paper_txt": paper_txt_path},
        "settings": {
            "backend": backend, "create_pr": create_pr, "repo_root": repo_root, "dry_run_pr": dry_run_pr,
            "goals_yaml": goals_yaml, "goal_key": goal_key, "gate_pr_on_goal": gate_pr_on_goal,
            "gate_min_score": gate_min_score, "gate_allow_unmet": gate_allow_unmet,
            "faithfulness_topk": faithfulness_topk, "skip_code": skip_code, "skip_text": skip_text,
            "seed": seed
        },
        "stages": {}
    }

    # ---------- CODE ----------
    code_result = None
    try:
        if not skip_code:
            stage_t0 = time.time()
            print("🔧 Improving code…")
            code_result = ci.improve(spec)
            code_action = vc.add_vpm_row(code_result["vpm_row"], f"code:{spec['function_name']}")
            kb.publish("decision.emitted", {"unit": f"code:{spec['function_name']}", "signal": code_action.signal.name, "reason": code_action.reason, "params": code_action.params})
            report["stages"]["code"] = {
                "ok": True,
                "elapsed_sec": round(time.time() - stage_t0, 2),
                "vpm_row": code_result["vpm_row"],
                "action": code_action,
                "artifacts": code_result.get("run_dir"),
                "passed": code_result.get("passed"),
                "edit_log": code_result.get("edit_log", []),
            }
            _append_ndjson(trace_path, {"ts": _ts(), "run_id": run_id, "stage": "code", "vpm": code_result["vpm_row"], "action": code_action})
            print(f"→ Code VPM: {code_result['vpm_row']}")
            print(f"→ Controller: {code_action}")
        else:
            report["stages"]["code"] = {"ok": True, "skipped": True}

    except Exception as e:
        tb = traceback.format_exc()
        report["stages"]["code"] = {"ok": False, "error": str(e), "traceback": tb}
        _append_ndjson(trace_path, {"ts": _ts(), "run_id": run_id, "stage": "code", "error": str(e)})
        print("❌ Code stage failed:", e)

    # ---------- TEXT ----------
    text_result = None
    try:
        if not skip_text:
            stage_t0 = time.time()
            print("📝 Improving text…")
            text_result = ti.improve(plan)
            text_action = vc.add_vpm_row(text_result["vpm_row"], f"text:{plan.get('section_title','section')}")
            kb.publish("decision.emitted", {"unit": f"text:{plan.get('section_title','section')}", "signal": text_action.signal.name, "reason": text_action.reason, "params": text_action.params})

            report["stages"]["text"] = {
                "ok": True,
                "elapsed_sec": round(time.time() - stage_t0, 2),
                "vpm_row": text_result["vpm_row"],
                "action": text_action,
                "artifacts": text_result.get("run_dir"),
                "passed": text_result.get("passed")
            }
            _append_ndjson(trace_path, {"ts": _ts(), "run_id": run_id, "stage": "text", "vpm": text_result["vpm_row"], "action": text_action})
            print(f"→ Text VPM: {text_result['vpm_row']}")
            print(f"→ Controller: {text_action}")
        else:
            report["stages"]["text"] = {"ok": True, "skipped": True}
    except Exception as e:
        tb = traceback.format_exc()
        report["stages"]["text"] = {"ok": False, "error": str(e), "traceback": tb}
        _append_ndjson(trace_path, {"ts": _ts(), "run_id": run_id, "stage": "text", "error": str(e)})
        print("❌ Text stage failed:", e)

    # ---------- GOALS & GATING ----------
    goal_eval = None
    try:
        kind, name = goal_key
        if text_result and text_result.get("vpm_row"):
            goal_eval = gs.score(kind, name, text_result["vpm_row"])
            report["stages"]["goals"] = {"ok": True, "goal": {"kind": kind, "name": name}, "eval": goal_eval}
            print(f"🎯 Goal [{kind}/{name}] Score: {goal_eval['score']:.3f} | Unmet: {goal_eval['unmet']}")
        else:
            report["stages"]["goals"] = {"ok": True, "skipped": True}
    except Exception as e:
        tb = traceback.format_exc()
        report["stages"]["goals"] = {"ok": False, "error": str(e), "traceback": tb}
        print("⚠️ Goal scoring failed:", e)

    # ---------- FAITHFULNESS (optional) ----------
    faith = None
    try:
        if paper_txt_path and text_result and text_result.get("vpm_row"):
            paper_text = Path(paper_txt_path).read_text()
            fb = FaithfulnessBot(top_k=faithfulness_topk)
            fb.prepare_paper(paper_text)
            # Expect text improver plan units with claim_id/claim; fallback to sampling bullets
            claims = plan.get("units") or []
            claims_payload = [{"claim_id": u.get("claim_id"), "claim": u.get("claim", "")} for u in claims][:10]
            faith_results = fb.verify_claims_batch(claims_payload)
            faith_score = fb.get_faithfulness_score(claims_payload)
            faith_stats = fb.get_stats(faith_results)
            faith = {"score": faith_score, "stats": faith_stats}
            report["stages"]["faithfulness"] = {"ok": True, **faith}
            print(f"🧪 Faithfulness: {faith_score:.3f} (supported {faith_stats['supported']}/{faith_stats['total']})")
        else:
            report["stages"]["faithfulness"] = {"ok": True, "skipped": True}
    except Exception as e:
        tb = traceback.format_exc()
        report["stages"]["faithfulness"] = {"ok": False, "error": str(e), "traceback": tb}
        print("⚠️ Faithfulness stage failed:", e)

    # ---------- PR (optional, gated) ----------
    try:
        if create_pr:
            print("🚀 PR phase…")
            rl = RepoLink(repo_root=repo_root, dry_run=dry_run_pr)

            def _gate(ok_flag: bool, vpm: Dict[str, Any], goal: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
                if not gate_pr_on_goal or (not goal):
                    return ok_flag, "no_goal_gate"
                score = goal["score"]
                unmet = goal.get("unmet", [])
                if score >= gate_min_score and len(unmet) <= gate_allow_unmet:
                    return ok_flag, "goal_pass"
                return False, f"goal_block(score={score:.3f}, unmet={len(unmet)})"

            # Code PR
            code_pr = {"attempted": False}
            if code_result and code_result.get("passed"):
                ok, reason = _gate(True, code_result["vpm_row"], None)  # code not goal-gated by default
                if ok:
                    code_pr_url = rl.create_pr(code_result["run_dir"], code_result["vpm_row"], "code")
                    code_pr = {"attempted": True, "ok": True, "url": code_pr_url}
                else:
                    code_pr = {"attempted": True, "ok": False, "reason": reason}

            # Text PR (goal-gated)
            text_pr = {"attempted": False}
            if text_result and text_result.get("passed"):
                ok, reason = _gate(True, text_result["vpm_row"], goal_eval)
                if ok:
                    text_pr_url = rl.create_pr(text_result["run_dir"], text_result["vpm_row"], "text")
                    text_pr = {"attempted": True, "ok": True, "url": text_pr_url}
                else:
                    text_pr = {"attempted": True, "ok": False, "reason": reason}

            report["stages"]["pr"] = {"ok": True, "code": code_pr, "text": text_pr, "dry_run": dry_run_pr}
        else:
            report["stages"]["pr"] = {"ok": True, "skipped": True}
    except Exception as e:
        tb = traceback.format_exc()
        report["stages"]["pr"] = {"ok": False, "error": str(e), "traceback": tb}
        print("⚠️ PR stage failed:", e)

    # ---------- finalize ----------
    report["finished"] = _ts()
    report["elapsed_sec"] = round(time.time() - t0, 2)
    _write_json(workdir_p / "report.json", report)
    print(f"✅ Report saved: {workdir_p / 'report.json'}")
    return report

# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Run paper section improver (code+text) with goals & optional PR gating")
    parser.add_argument("--spec", required=True, help="Path to spec.json")
    parser.add_argument("--plan", required=True, help="Path to plan.json")
    parser.add_argument("--workdir", default="./runs", help="Working directory")
    parser.add_argument("--backend", default="torch", choices=["torch", "numpy"], help="Code backend")
    parser.add_argument("--pr", action="store_true", help="Create PRs if passed")
    parser.add_argument("--repo-root", default="../..", help="Root of git repo")
    parser.add_argument("--dry-run-pr", action="store_true", help="Do not actually open PRs")
    parser.add_argument("--goals-yaml", default=None, help="Optional YAML to override goal templates")
    parser.add_argument("--goal-kind", default="text", choices=["text", "code"], help="Goal kind for gating")
    parser.add_argument("--goal-name", default="academic_summary", help="Goal name for gating")
    parser.add_argument("--gate-min-score", type=float, default=0.75, help="Minimum goal score to allow PR")
    parser.add_argument("--gate-allow-unmet", type=int, default=0, help="Allowed unmet bars for PR gating")
    parser.add_argument("--paper-txt", default=None, help="Optional: raw paper text to run faithfulness on")
    parser.add_argument("--faith-topk", type=int, default=5, help="Top-k passages for faithfulness judge")
    parser.add_argument("--skip-code", action="store_true", help="Skip code improver")
    parser.add_argument("--skip-text", action="store_true", help="Skip text improver")
    parser.add_argument("--seed", type=int, default=0, help="Global seed")
    args = parser.parse_args()

    run_paper_section(
        spec_path=args.spec,
        plan_path=args.plan,
        workdir=args.workdir,
        backend=args.backend,
        create_pr=args.pr,
        repo_root=args.repo_root,
        goals_yaml=args.goals_yaml,
        goal_key=(args.goal_kind, args.goal_name),
        gate_pr_on_goal=True,
        gate_min_score=args.gate_min_score,
        gate_allow_unmet=args.gate_allow_unmet,
        paper_txt_path=args.paper_txt,
        faithfulness_topk=args.faith_topk,
        skip_code=args.skip_code,
        skip_text=args.skip_text,
        dry_run_pr=args.dry_run_pr,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
