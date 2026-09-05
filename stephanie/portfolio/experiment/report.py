# stephanie/portfolio/experiment/report.py
"""Experiment 001 report (§Experiment 001: report I want at the end)."""
from __future__ import annotations

from typing import Mapping, Sequence

from stephanie.portfolio.experiment.arm import ExperimentArm
from stephanie.portfolio.experiment.metrics import ArmMetrics


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a (unknown)"
    return f"{value:.{digits}f}"


def render_report(
    title: str,
    arm_metrics: Mapping[ExperimentArm, ArmMetrics],
    task_types: Sequence[str],
    notes: Sequence[str] = (),
) -> str:
    lines = [title, "=" * len(title), ""]
    total_runs = sum(m.runs for m in arm_metrics.values())
    lines.append(f"Runs:           {total_runs}")
    lines.append(f"Task families:  {', '.join(task_types)}")
    lines.append("")

    def section(name: str) -> None:
        lines.append(name)
        lines.append("-" * 34)

    baseline = arm_metrics.get(ExperimentArm.A_PRIMARY_ONLY)
    base_unique = baseline.unique_valid_issue_count if baseline else 0

    section("UNIQUE VALID FINDINGS")
    for arm, metrics in arm_metrics.items():
        delta = metrics.unique_valid_issue_count - base_unique if arm != ExperimentArm.A_PRIMARY_ONLY else 0
        suffix = f"   +{delta}" if arm != ExperimentArm.A_PRIMARY_ONLY else "  (baseline)"
        lines.append(f"{arm.value}  {metrics.arm.name:<22} {metrics.unique_valid_issue_count:<5} {suffix}")
    lines.append("")

    section("VALID ISSUE COUNT (incl. cross-case duplicates)")
    for arm, metrics in arm_metrics.items():
        lines.append(f"{arm.value}  {metrics.arm.name:<22} {metrics.valid_issue_count}")
    lines.append("")

    section("PRIMARY ERRORS CAUGHT (TPs arm found that A missed)")
    for arm, metrics in arm_metrics.items():
        if arm == ExperimentArm.A_PRIMARY_ONLY:
            continue
        lines.append(f"{arm.value}  {metrics.arm.name:<22} {metrics.primary_errors_caught}")
    lines.append("")

    section("FALSE POSITIVES / UNVERIFIABLE")
    for arm, metrics in arm_metrics.items():
        lines.append(
            f"{arm.value}  {metrics.arm.name:<22} fp={metrics.false_positive_count} "
            f"fp_rate={_fmt(metrics.false_positive_rate)} unverifiable={metrics.unverifiable_count}"
        )
    lines.append("")

    section("EFFICIENCY (local models: tokens + latency, $0)")
    for arm, metrics in arm_metrics.items():
        lines.append(
            f"{arm.value}  {metrics.arm.name:<22} in={metrics.input_tokens} out={metrics.output_tokens} "
            f"latency_s={metrics.latency_ms / 1000:.1f} unique_per_Mtok={_fmt(metrics.unique_per_million_tokens, 1)}"
        )
    lines.append("")

    if notes:
        section("NOTES / LIMITATIONS")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    return "\n".join(lines)
