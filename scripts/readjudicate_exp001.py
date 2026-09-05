"""Re-adjudicate saved Experiment 001 runs (no model calls).

Replays raw model outputs from runs.jsonl through the current
DeterministicAdjudicator + metrics, for adjudication-rule iterations.

Usage:
    python scripts/readjudicate_exp001.py --in outputs/portfolio_exp001_canary --out outputs/portfolio_exp001_readjudicated
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_portfolio_experiment_001 import build_cases, parse_findings

from stephanie.portfolio.experiment import (
    DeterministicAdjudicator,
    ExperimentArm,
    FindingClass,
    PortfolioExperimentRun,
    compute_arm_metrics,
    render_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    args = parser.parse_args()

    in_dir = Path(args.inp)
    records = [json.loads(line) for line in open(in_dir / "runs.jsonl", encoding="utf-8")]
    case_ids = sorted({r["case_id"] for r in records})
    chapters = [c.split("-")[1] for c in case_ids]
    cases = {c.case_id: c for c in build_cases(chapters, 3500)}
    adjudicator = DeterministicAdjudicator()

    runs: list[PortfolioExperimentRun] = []
    for record in records:
        case = cases[record["case_id"]]
        arm = ExperimentArm(record["arm"])
        findings = []
        for candidate_id, raw in (record.get("raw_outputs") or {}).items():
            for finding in parse_findings(raw, case.case_id, arm.value, candidate_id):
                findings.extend(f for f, _ in adjudicator.adjudicate([finding], case))
        in_tokens, out_tokens = record["tokens"]
        runs.append(PortfolioExperimentRun(
            run_id=record["run_id"], case_id=record["case_id"],
            task_type="book.argument.review", arm=arm, repetition=0,
            primary_request_id=record.get("primary_request_id", ""),
            candidate_ids=tuple((record.get("raw_outputs") or {}).keys()),
            findings=tuple(findings),
            input_tokens=in_tokens, output_tokens=out_tokens,
            latency_ms=record.get("latency_ms", 0.0),
        ))

    baseline_tp = {}
    for run in runs:
        if run.arm == ExperimentArm.A_PRIMARY_ONLY:
            baseline_tp[run.case_id] = {
                f.matched_code for f in run.findings
                if f.classification == FindingClass.TRUE_POSITIVE and f.matched_code
            }
    arm_metrics = {arm: compute_arm_metrics(
        arm, [r for r in runs if r.arm == arm], baseline_tp) for arm in ExperimentArm}

    overlap_lines = []
    for arm in (ExperimentArm.B_SAME_FAMILY_CRITIC, ExperimentArm.C_FRONTIER_REVIEWER,
                ExperimentArm.D_BREADTH):
        missed_a = missed_x = shared = total = 0
        for case_id in case_ids:
            case = cases[case_id]
            case_codes = {e.code for e in case.expected}
            a_run = next((r for r in runs if r.case_id == case_id and r.arm == ExperimentArm.A_PRIMARY_ONLY), None)
            x_run = next((r for r in runs if r.case_id == case_id and r.arm == arm), None)
            if a_run is None or x_run is None:
                continue
            a_missed = case_codes - {f.matched_code for f in a_run.findings
                                     if f.classification == FindingClass.TRUE_POSITIVE}
            x_missed = case_codes - {f.matched_code for f in x_run.findings
                                     if f.classification == FindingClass.TRUE_POSITIVE}
            missed_a += len(a_missed)
            missed_x += len(x_missed)
            shared += len(a_missed & x_missed)
            total += len(case_codes)
        overlap = (shared / missed_a) if missed_a else None
        overlap_lines.append(
            f"{arm.value} missed_given_A_missed={overlap if overlap is None else round(overlap, 2)} "
            f"(A missed {missed_a}/{total}, {arm.value} missed {missed_x}/{total})")

    notes = [
        "Re-adjudication with content-evidence rule (code label alone is not a detection).",
        "Canary: 1 repetition; stochastic variance not measured; temperature 0.",
        "Adjudication deterministic only; unmatched findings UNVERIFIABLE, never LLM-judged.",
        "Corpus caveat: draft defects in the frozen YAML were fixed in current chapter text; "
        "recall is measured against defect descriptions, so near-zero recall is expected, not a model verdict.",
        "Local models: $0 cost; efficiency as tokens + latency (MUI/$ undefined).",
    ]
    report = render_report("Stephanie Portfolio Experiment 001 (canary, re-adjudicated)",
                           arm_metrics, ["book.argument.review"], notes)
    report += "\nFAILURE OVERLAP WITH PRIMARY (missed-given-A-missed)\n----------------------------------\n"
    report += "\n".join(overlap_lines) + "\n"
    print(report)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    with open(out_dir / "runs.jsonl", "w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps({
                "run_id": run.run_id, "arm": run.arm.value, "case_id": run.case_id,
                "findings": [{"claim": f.claim[:200], "classification": f.classification.value,
                              "matched_code": f.matched_code} for f in run.findings],
            }) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
