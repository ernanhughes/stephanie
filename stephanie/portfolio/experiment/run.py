# stephanie/portfolio/experiment/run.py
"""Experiment run records (§Experiment 001: data model)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional

from stephanie.portfolio.experiment.arm import ExperimentArm
from stephanie.portfolio.experiment.finding import Finding


@dataclass(frozen=True)
class PortfolioExperimentRun:
    run_id: str
    case_id: str
    task_type: str
    arm: ExperimentArm
    repetition: int

    primary_request_id: str  # A-identity: same primary leg across arms
    candidate_ids: tuple[str, ...] = ()
    findings: tuple[Finding, ...] = ()

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioExperiment:
    experiment_id: str
    cases: list = field(default_factory=list)
    runs: list[PortfolioExperimentRun] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def runs_for(self, arm: ExperimentArm) -> list[PortfolioExperimentRun]:
        return [r for r in self.runs if r.arm == arm]
