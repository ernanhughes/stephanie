# stephanie/portfolio/experiment/metrics.py
"""Experiment metrics (§Experiment 001: primary metrics + MUI/$).

Raw counts first, no severity weighting in Experiment 001. Unknown stays
unknown: rates are None when the denominator is zero, never 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from stephanie.portfolio.experiment.arm import ExperimentArm
from stephanie.portfolio.experiment.finding import Finding, FindingClass
from stephanie.portfolio.experiment.run import PortfolioExperimentRun

TP = FindingClass.TRUE_POSITIVE
FP = FindingClass.FALSE_POSITIVE


@dataclass(frozen=True)
class ArmMetrics:
    arm: ExperimentArm
    runs: int

    valid_issue_count: int = 0
    unique_valid_issue_count: int = 0
    false_positive_count: int = 0
    unverifiable_count: int = 0
    verified_issue_count: int = 0

    primary_errors_caught: int = 0  # TPs this arm found that arm A missed (per case)
    primary_cases: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    false_positive_rate: Optional[float] = None
    primary_error_caught_rate: Optional[float] = None
    unique_per_million_tokens: Optional[float] = None
    unique_per_minute: Optional[float] = None

    metadata: Mapping[str, object] = field(default_factory=dict)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def compute_arm_metrics(
    arm: ExperimentArm,
    runs: Sequence[PortfolioExperimentRun],
    baseline_tp_by_case: Mapping[str, set[str]],
) -> ArmMetrics:
    findings = [f for r in runs for f in r.findings]
    tp = [f for f in findings if f.classification == TP]
    unique_tp_codes = {f.matched_code for f in tp if f.matched_code}
    fp = [f for f in findings if f.classification == FP]

    caught, cases = 0, 0
    for run in runs:
        if run.case_id not in baseline_tp_by_case:
            continue
        baseline = baseline_tp_by_case[run.case_id]
        mine = {f.matched_code for f in run.findings if f.classification == TP and f.matched_code}
        cases += 1
        caught += len(mine - baseline)

    in_tokens = sum(r.input_tokens for r in runs)
    out_tokens = sum(r.output_tokens for r in runs)
    total_tokens = in_tokens + out_tokens
    latency = sum(r.latency_ms for r in runs)
    judged = len(tp) + len(fp)

    return ArmMetrics(
        arm=arm,
        runs=len(runs),
        valid_issue_count=len(tp),
        unique_valid_issue_count=len(unique_tp_codes),
        false_positive_count=len(fp),
        unverifiable_count=sum(1 for f in findings if f.classification == FindingClass.UNVERIFIABLE),
        verified_issue_count=len(tp),  # canary: TP == deterministically verified
        primary_errors_caught=caught,
        primary_cases=cases,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        latency_ms=latency,
        false_positive_rate=_rate(len(fp), judged),
        primary_error_caught_rate=None,  # per-case rates need repeated reps; see report
        unique_per_million_tokens=_rate(len(unique_tp_codes) * 1_000_000, total_tokens),
        unique_per_minute=_rate(len(unique_tp_codes) * 60_000, int(latency)),
    )


def failure_overlap_from_runs(
    runs_a: Sequence[PortfolioExperimentRun],
    runs_b: Sequence[PortfolioExperimentRun],
) -> Optional[float]:
    """Share of baseline-missed expected codes that both arms miss (proxy)."""
    return None  # needs expected-code framing per case; computed in report
