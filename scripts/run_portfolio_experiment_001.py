"""Experiment 001 canary runner (Stage 3.8A).

Corpus: frozen Writer argument-eval diagnostics (chapters 07-11) + chapter
excerpts from new-books. Ground truth: TP codes per chapter + known FP
traps (ORPHAN_SECTION lexical, WEAK_TRANSITION checker artifact).

Arms share ONE primary ModelRequest per case (A-identity). Deterministic
planner only (use_historical_performance stays False). Temperature 0.

Usage:
    python scripts/run_portfolio_experiment_001.py --limit-cases 2
    python scripts/run_portfolio_experiment_001.py --cases 07,08,09,10,11 --reps 1
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stephanie.evaluation import Criterion, EvaluatorRef, ScoreScale
from stephanie.evaluation.evaluation import EvaluationObservation
from stephanie.evaluation.score import Score
from stephanie.evaluation.subject import SubjectRef
from stephanie.models import Model as RuntimeModel
from stephanie.models import ModelRequest
from stephanie.portfolio import PortfolioPlanner, PortfolioPolicy, PortfolioRole
from stephanie.portfolio.executor import PortfolioExecutor
from stephanie.portfolio.experiment import (
    DeterministicAdjudicator,
    ExpectedFinding,
    ExperimentArm,
    Finding,
    FindingClass,
    PortfolioBenchmarkCase,
    PortfolioExperiment,
    PortfolioExperimentRun,
    compute_arm_metrics,
    render_report,
)
from stephanie.portfolio.experiment.ollama_provider import OllamaChatProvider
from stephanie.services.evaluation_runtime import EvaluationRuntime
from stephanie.services.model_runtime import ModelRuntime

NEW_BOOKS = Path(r"C:\Projects\new-books\content\books\dspy-from-first-principles")
EVAL_YAML = Path(r"C:\Projects\writer\evals\argument-eval-v1-dspy-07-11.yaml")

PRIMARY_MODEL = "ollama:batiai/qwen3.6-27b:q4"
CRITIC_MODEL = "ollama:qwen3:latest"
REVIEWER_MODEL = "ollama:qwen3.6:latest"
BREADTH_MODEL = "ollama:mistral:7b-instruct"

TASK_TYPE = "book.argument.review"
REVIEW_PROMPT = """Review the following book chapter excerpt for ARGUMENT-STRUCTURE defects only.
Report each finding as one JSON object with keys: category, claim, location.
Allowed categories: DUPLICATE_ID, UNKNOWN_CONCEPT, CONCEPT_BEFORE_DEFINITION,
ORPHAN_CONCEPT, UNUSED_EVIDENCE, ORPHAN_SECTION, WEAK_TRANSITION, OTHER.
Output ONLY the JSON array, no preamble and no explanation.

Excerpt:
""".strip()


# ---------------------------------------------------------------- corpus

def parse_expected_from_yaml(text: str) -> tuple[dict[str, list], set[str]]:
    """Regex-parse diagnostic blocks (YAML has a pre-existing syntax error
    outside the diagnostics section, so no YAML parser)."""
    per_chapter: dict[str, list] = {}
    fp_traps: set[str] = {"ORPHAN_SECTION", "WEAK_TRANSITION"}
    blocks = re.split(r"(?m)^\s+-\s+(?=chapter:|chapters:|transition:)", text)
    for block in blocks:
        chapter_match = re.search(r"chapter:\s*(\d+)", block)
        code_match = re.search(r"code:\s*([A-Z_]+)", block)
        class_match = re.search(r"classification:\s*(TP|FP)", block)
        note_match = re.search(r"note:\s*(.+?)(?=\n\s+\w+:|\Z)", block, re.S)
        if not chapter_match or not code_match or not class_match:
            continue
        if class_match.group(1) != "TP":
            continue
        code = code_match.group(1)
        note = " ".join(note_match.group(1).split()) if note_match else ""
        keywords = tuple(w for w in re.findall(r"[a-z][a-z0-9_-]{3,}", note.lower())[:8])
        per_chapter.setdefault(chapter_match.group(1).zfill(2), []).append(
            ExpectedFinding(code=code, keywords=keywords, note=note[:200])
        )
    return per_chapter, fp_traps


def build_cases(chapters: list[str], max_chars: int) -> list[PortfolioBenchmarkCase]:
    yaml_text = EVAL_YAML.read_text(encoding="utf-8")
    per_chapter, fp_traps = parse_expected_from_yaml(yaml_text)
    cases = []
    for chapter in chapters:
        chapter_file = NEW_BOOKS / f"{chapter}-chapter.md"
        if not chapter_file.exists():
            print(f"SKIP {chapter}: {chapter_file} missing")
            continue
        excerpt = chapter_file.read_text(encoding="utf-8")[:max_chars]
        prompt = REVIEW_PROMPT + "\n" + excerpt
        cases.append(PortfolioBenchmarkCase.freeze(
            case_id=f"argreview-{chapter}",
            task_type=TASK_TYPE,
            prompt=prompt,
            source_text=excerpt,
            expected=tuple(per_chapter.get(chapter, [])),
            metadata={"chapter": chapter, "known_fp": sorted(fp_traps)},
        ))
    return cases


# ---------------------------------------------------------------- models

def check_models(required: list[str]) -> None:
    request = urllib.request.Request("http://localhost:11434/api/tags")
    with urllib.request.urlopen(request, timeout=10) as response:
        available = {m["name"] for m in json.loads(response.read().decode()).get("models", [])}
    missing = []
    for ref in required:
        name = ref.split(":", 1)[1] if ":" in ref else ref
        if name not in available and not any(name in a or a in name for a in available):
            missing.append(ref)
    if missing:
        print(f"WARNING: models not in Ollama library (will fail fast per call): {missing}")
    else:
        print(f"models OK: {required}")


def build_runtime(num_predict: int) -> ModelRuntime:
    runtime = ModelRuntime()
    provider = OllamaChatProvider(num_predict=num_predict)
    runtime.register_provider("ollama", provider)
    for ref in (PRIMARY_MODEL, CRITIC_MODEL, REVIEWER_MODEL, BREADTH_MODEL):
        runtime.register_model(RuntimeModel.from_ref(ref))
    return runtime


# ---------------------------------------------------------------- findings

def _with_model(candidate, model_ref: str):
    """Pin a planned candidate to an explicit experiment model."""
    import dataclasses

    model = RuntimeModel.from_ref(model_ref)
    return dataclasses.replace(
        candidate, model_id=model.id,
        request=dataclasses.replace(candidate.request, model=model),
        metadata={**candidate.metadata, "reason": f"experiment pin ({model.id})"},
    )


def parse_findings(output: str, case_id: str, arm: str, candidate_id: str) -> list[Finding]:
    items = _parse_json_array(output)
    if items is None:
        items = _salvage_objects(output)
    if not items:
        items = [{"category": "OTHER", "claim": output[:500], "location": None}]
    findings = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        findings.append(Finding(
            finding_id=f"{case_id}-{arm}-{candidate_id}-f{i}",
            case_id=case_id, arm=arm, candidate_id=candidate_id,
            category=str(item.get("category", "OTHER")),
            claim=str(item.get("claim", ""))[:1000],
            location=str(item.get("location")) if item.get("location") else None,
        ))
    return findings


def _parse_json_array(output: str):
    try:
        start, end = output.index("["), output.rindex("]") + 1
        items = json.loads(output[start:end])
        return items if isinstance(items, list) else None
    except (ValueError, json.JSONDecodeError):
        return None


def _salvage_objects(output: str) -> list:
    """Recover complete {...} objects from truncated JSON (partial credit
    for findings actually emitted before the token cap)."""
    decoder = json.JSONDecoder()
    items: list = []
    idx = 0
    while idx < len(output):
        start = output.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(output, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            items.append(obj)
        idx = end
    return items


# ---------------------------------------------------------------- main

async def run_case(case, runtime, eval_runtime, planner, adjudicator, rep: int,
                   out_records: list) -> list[PortfolioExperimentRun]:
    # ONE frozen primary request shared by every arm (A-identity).
    primary_request = ModelRequest(
        model=PRIMARY_MODEL, prompt=case.prompt, task_type=case.task_type,
        trace_id=f"{case.case_id}-primary-r{rep}",
    )
    policy = PortfolioPolicy(task_type=case.task_type,
                             require_different_provider_for_reviewer=False)
    plan = await planner.plan(primary_request, policy)
    by_role = {c.role: c for c in plan.candidates}
    # Explicit arm models: local bench shares one provider, so independence
    # comes from different weights/families, not provider strings.
    by_role[PortfolioRole.INDEPENDENT_REVIEWER] = _with_model(
        by_role[PortfolioRole.INDEPENDENT_REVIEWER], REVIEWER_MODEL)
    by_role[PortfolioRole.BREADTH] = _with_model(
        by_role[PortfolioRole.BREADTH], BREADTH_MODEL)
    executor = PortfolioExecutor(runtime)

    async def invoke(candidate):
        executions = await executor.execute(
            __import__("stephanie.portfolio.plan", fromlist=["PortfolioPlan"]).PortfolioPlan(
                plan_id=plan.plan_id, task_type=plan.task_type, candidates=(candidate,),
                budget=plan.budget, synthesis_policy=plan.synthesis_policy,
                created_at=plan.created_at,
            )
        )
        return executions[0]

    # Arm A: primary only.
    primary_exec = await invoke(by_role[PortfolioRole.PRIMARY])
    arm_executions = {
        ExperimentArm.A_PRIMARY_ONLY: [primary_exec],
        ExperimentArm.C_FRONTIER_REVIEWER: [primary_exec, await invoke(by_role[PortfolioRole.INDEPENDENT_REVIEWER])],
        ExperimentArm.D_BREADTH: [primary_exec, await invoke(by_role[PortfolioRole.BREADTH])],
    }
    # Arm B: same primary leg + anchored same-family critic.
    import dataclasses

    from stephanie.portfolio import PortfolioCandidate, PortfolioRole as Role

    critic_candidate = PortfolioCandidate(
        candidate_id=f"critic_{case.case_id}_r{rep}",
        model_id=CRITIC_MODEL, role=Role.CRITIC,
        request=dataclasses.replace(primary_request, model=RuntimeModel.from_ref(CRITIC_MODEL)),
        independence_group="anchored_critique",
    )
    critic_exec = await executor.execute_anchored(
        critic_candidate, [primary_exec], [primary_exec.candidate_id])
    arm_executions[ExperimentArm.B_SAME_FAMILY_CRITIC] = [primary_exec, critic_exec]
    # Arm E: everything + synthesis selection over all executions.
    arm_executions[ExperimentArm.E_FULL] = [
        primary_exec, critic_exec,
        arm_executions[ExperimentArm.C_FRONTIER_REVIEWER][1],
        arm_executions[ExperimentArm.D_BREADTH][1],
    ]

    # Arm-specific reviewer/breadth model override: planner picks from registry;
    # force the intended models for C and D.
    runs = []
    for arm, executions in arm_executions.items():
        findings: list[Finding] = []
        in_tokens = out_tokens = 0
        latency = 0.0
        for execution in executions:
            in_tokens += (execution.usage.input_tokens or 0) if execution.usage else 0
            out_tokens += (execution.usage.output_tokens or 0) if execution.usage else 0
            latency += execution.latency_ms or 0.0
            for finding in parse_findings(execution.output_text, case.case_id, arm.value, execution.candidate_id):
                findings.append(finding)
        adjudicated = [f for finding in findings for f, _ in adjudicator.adjudicate([finding], case)]
        # Record per-code recall evaluations (Stage 2 linkage).
        for execution in executions:
            exec_findings = [f for f in adjudicated if f.candidate_id == execution.candidate_id]
            tp_codes = {f.matched_code for f in exec_findings if f.classification == FindingClass.TRUE_POSITIVE}
            observation = EvaluationObservation(
                subject=SubjectRef(subject_type="model.response",
                                   subject_id=execution.request_id or execution.execution_id,
                                   text=execution.output_text[:2000]),
                criterion=Criterion(name="argument_review_recall", scale=ScoreScale(0.0, 1.0)),
                evaluator=__import__("stephanie.evaluation", fromlist=["EvaluatorRef"]).EvaluatorRef(name="exp001_deterministic"),
                model_id=execution.model_id, task_type=case.task_type,
                scores=[Score(score_id="", evaluation_id="", dimension=code,
                              value=1.0 if code in tp_codes else 0.0, scorer="exp001")
                        for code in sorted({e.code for e in case.expected})],
                metadata={"portfolio_candidate_id": execution.candidate_id,
                          "portfolio_role": execution.role.value,
                          "experiment_arm": arm.value, "case_id": case.case_id},
            )
            await eval_runtime.record_from_model_invocation(
                observation, model_id=execution.model_id, request_id=execution.request_id,
                trace_id=execution.trace_id, task_type=case.task_type)
        run = PortfolioExperimentRun(
            run_id=f"{case.case_id}-{arm.value}-r{rep}", case_id=case.case_id,
            task_type=case.task_type, arm=arm, repetition=rep,
            primary_request_id=primary_exec.request_id,
            candidate_ids=tuple(e.candidate_id for e in executions),
            findings=tuple(adjudicated),
            input_tokens=in_tokens, output_tokens=out_tokens, latency_ms=latency,
            metadata={"plan_id": plan.plan_id,
                      "raw_outputs": {e.candidate_id: e.output_text[:1500] for e in executions}},
        )
        runs.append(run)
        out_records.append({
            "run_id": run.run_id, "arm": arm.value, "case_id": case.case_id,
            "primary_request_id": run.primary_request_id,
            "candidate_roles": {e.candidate_id: e.role.value for e in executions},
            "findings": [{"candidate_id": f.candidate_id, "claim": f.claim[:200],
                          "classification": f.classification.value,
                          "matched_code": f.matched_code} for f in adjudicated],
            "raw_outputs": run.metadata.get("raw_outputs", {}),
            "tokens": [in_tokens, out_tokens], "latency_ms": latency,
        })
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="07,08,09,10,11")
    parser.add_argument("--corpus-dir", default="")
    parser.add_argument("--case-filter", default="")
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=3500)
    parser.add_argument("--num-predict", type=int, default=192)
    parser.add_argument("--out", default="outputs/portfolio_exp001")
    args = parser.parse_args()

    corpus_meta = None
    if args.corpus_dir:
        from stephanie.portfolio.experiment.corpus import load_corpus

        cases, corpus_meta = load_corpus(args.corpus_dir)
        if args.case_filter:
            keys = [k.strip() for k in args.case_filter.split(",") if k.strip()]
            cases = [c for c in cases if any(
                k in c.case_id or k == (c.metadata or {}).get("pair_id") for k in keys)]
        elif args.limit_cases:
            # Keep pairs intact: first N pairs (defect + twin) in manifest order.
            wanted: list[str] = []
            for case in cases:
                pair = (case.metadata or {}).get("pair_id", case.case_id)
                if pair not in wanted:
                    wanted.append(pair)
            wanted = wanted[: args.limit_cases]
            cases = [c for c in cases
                     if (c.metadata or {}).get("pair_id", c.case_id) in wanted]
    else:
        chapters = [c.strip() for c in args.cases.split(",") if c.strip()]
        if args.limit_cases:
            chapters = chapters[: args.limit_cases]
        cases = build_cases(chapters, args.max_chars)
    print(f"cases: {[c.case_id for c in cases]}")
    for case in cases:
        print(f"  {case.case_id}: {len(case.expected)} expected TP codes, hash={case.source_hash}")
    check_models([PRIMARY_MODEL, CRITIC_MODEL, REVIEWER_MODEL, BREADTH_MODEL])

    async def _main():
        runtime = build_runtime(args.num_predict)
        eval_runtime = EvaluationRuntime()
        models = [RuntimeModel.from_ref(PRIMARY_MODEL),
                  RuntimeModel.from_ref(REVIEWER_MODEL),
                  RuntimeModel.from_ref(BREADTH_MODEL),
                  RuntimeModel.from_ref(CRITIC_MODEL)]
        planner = PortfolioPlanner(models)
        adjudicator = DeterministicAdjudicator()
        experiment = PortfolioExperiment(experiment_id="exp001", cases=cases)
        out_records: list = []
        for case in cases:
            for rep in range(args.reps):
                print(f"--- {case.case_id} rep {rep} ---", flush=True)
                started = time.time()
                runs = await run_case(case, runtime, eval_runtime, planner, adjudicator, rep, out_records)
                experiment.runs.extend(runs)
                print(f"    done in {time.time() - started:.0f}s", flush=True)
        return experiment, out_records

    experiment, out_records = asyncio.run(_main())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs.jsonl", "w", encoding="utf-8") as handle:
        for record in out_records:
            handle.write(json.dumps(record) + "\n")

    # Metrics: baseline TP sets from arm A per case.
    baseline_tp: dict[str, set[str]] = {}
    for run in experiment.runs:
        if run.arm == ExperimentArm.A_PRIMARY_ONLY:
            baseline_tp[run.case_id] = {
                f.matched_code for f in run.findings
                if f.classification == FindingClass.TRUE_POSITIVE and f.matched_code
            }
    arm_metrics = {}
    for arm in ExperimentArm:
        arm_metrics[arm] = compute_arm_metrics(
            arm, [r for r in experiment.runs if r.arm == arm], baseline_tp)

    # Failure overlap with primary: share of A-missed codes also missed by X.
    overlap_lines = []
    for arm in (ExperimentArm.B_SAME_FAMILY_CRITIC, ExperimentArm.C_FRONTIER_REVIEWER, ExperimentArm.D_BREADTH):
        missed_a = missed_x = shared = total = 0
        for run in experiment.runs:
            if run.arm != ExperimentArm.A_PRIMARY_ONLY:
                continue
            case_codes = {e.code for e in next(c for c in cases if c.case_id == run.case_id).expected}
            a_tp = {f.matched_code for f in run.findings if f.classification == FindingClass.TRUE_POSITIVE}
            a_missed = case_codes - a_tp
            x_run = next((r for r in experiment.runs
                          if r.case_id == run.case_id and r.arm == arm), None)
            if x_run is None:
                continue
            x_tp = {f.matched_code for f in x_run.findings if f.classification == FindingClass.TRUE_POSITIVE}
            x_missed = case_codes - x_tp
            missed_a += len(a_missed)
            missed_x += len(x_missed)
            shared += len(a_missed & x_missed)
            total += len(case_codes)
        overlap = (shared / missed_a) if missed_a else None
        overlap_lines.append(
            f"{arm.value} missed_given_A_missed={overlap if overlap is None else round(overlap, 2)} "
            f"(A missed {missed_a}/{total}, {arm.value} missed {missed_x}/{total})")

    notes = [
        "Canary: 1 repetition; stochastic variance not measured.",
        "Temperature 0 for arm fairness; A primary leg is the identical request object in B-E.",
        "Adjudication is deterministic only (frozen TP codes + FP-trap list); unmatched findings are UNVERIFIABLE, never LLM-judged.",
        "Local models: $0 cost; efficiency reported as tokens + latency (MUI/$ undefined).",
        "C arm uses qwen3.6 36B as the strongest available local reviewer (no frontier API in this environment).",
    ]
    report = render_report("Stephanie Portfolio Experiment 001 (canary)", arm_metrics,
                           sorted({c.task_type for c in cases}), notes)
    report += "\nFAILURE OVERLAP WITH PRIMARY (missed-given-A-missed)\n----------------------------------\n"
    report += "\n".join(overlap_lines) + "\n"

    status = {
        "experiment": "exp001",
        "corpus_version": (corpus_meta or {}).get("version", "legacy-chapters"),
        "adjudication_version": __import__(
            "stephanie.portfolio.experiment.adjudication",
            fromlist=["ADJUDICATION_VERSION"]).ADJUDICATION_VERSION,
        "reps": args.reps,
        "runs": len(experiment.runs),
        "status": "CANARY",
        "gates": {},
    }
    if corpus_meta is not None:
        from stephanie.portfolio.experiment.paired import (
            paired_analysis,
            render_paired_section,
            verdict,
        )

        analysis = paired_analysis(
            [{"run_id": r["run_id"], "case_id": r["case_id"], "arm": r["arm"],
              "findings": r["findings"], "raw_outputs": r.get("raw_outputs", {}),
              "primary_request_id": r.get("primary_request_id", ""),
              "candidate_roles": r.get("candidate_roles", {})} for r in out_records],
            cases,
        )
        report += "\n" + render_paired_section(analysis) + "\n"
        verdict_text, reasons = verdict(analysis)
        status["status"] = "VALIDATED" if verdict_text == "VALIDATED" else "INVALID"
        status["gates"] = {"verdict": verdict_text, "reasons": reasons,
                           "dd": analysis["dd"], "recall": analysis["recall"],
                           "clean_fp": analysis["clean_fp"]}
        report += f"\nCORPUS VERDICT: {verdict_text}\n"
        for reason in reasons:
            report += f"  - {reason}\n"
    print()
    print(report)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    (out_dir / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
