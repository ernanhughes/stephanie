# stephanie/portfolio/experiment/__init__.py
"""Portfolio benchmark harness (Stage 3.8A). Narrow by design; migrates to
Stage 4 Experiment Runtime later."""
from __future__ import annotations

from stephanie.portfolio.experiment.adjudication import (
    ADJUDICATION_VERSION,
    Adjudication,
    DeterministicAdjudicator,
)
from stephanie.portfolio.experiment.arm import ARM_PURPOSE, ExperimentArm
from stephanie.portfolio.experiment.case import ExpectedFinding, PortfolioBenchmarkCase
from stephanie.portfolio.experiment.corpus import load_corpus
from stephanie.portfolio.experiment.finding import Finding, FindingClass
from stephanie.portfolio.experiment.metrics import ArmMetrics, compute_arm_metrics
from stephanie.portfolio.experiment.ollama_provider import OllamaChatProvider
from stephanie.portfolio.experiment.report import render_report
from stephanie.portfolio.experiment.run import PortfolioExperiment, PortfolioExperimentRun

__all__ = [
    "ADJUDICATION_VERSION", "Adjudication", "DeterministicAdjudicator",
    "ARM_PURPOSE", "ExperimentArm",
    "ExpectedFinding", "PortfolioBenchmarkCase", "load_corpus",
    "Finding", "FindingClass",
    "ArmMetrics", "compute_arm_metrics",
    "OllamaChatProvider",
    "render_report",
    "PortfolioExperiment", "PortfolioExperimentRun",
]
