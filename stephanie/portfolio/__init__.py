# stephanie/portfolio/__init__.py
"""Stephanie Model Portfolio Runtime (Stage 3).

Independent intelligence first: use another model if and only if it is
expected to contribute useful independent information.
"""
from __future__ import annotations

from stephanie.portfolio.candidate import PortfolioCandidate
from stephanie.portfolio.diagnostics import (
    BUDGET_EXCEEDED,
    CORRELATED_FAILURE,
    DISAGREEMENT_UNRESOLVED,
    EVALUATION_MISSING,
    EXECUTION_FAILED,
    INDEPENDENCE_VIOLATION,
    NO_ELIGIBLE_MODEL,
    ROLE_UNFILLED,
    SYNTHESIS_FAILED,
    VERIFICATION_FAILED,
    ZERO_MARGINAL_GAIN,
    PortfolioDiagnostic,
)
from stephanie.portfolio.disagreement import (
    ASSUMPTION,
    CONFIDENCE,
    FACTUAL,
    INTERPRETATION,
    METHOD,
    MISSING_INFORMATION,
    RECOMMENDATION,
    Disagreement,
    DisagreementAnalyzer,
)
from stephanie.portfolio.evaluator import PortfolioEvaluator
from stephanie.portfolio.executor import PortfolioExecution, PortfolioExecutor
from stephanie.portfolio.history import ModelRolePerformance, role_performance
from stephanie.portfolio.independence import (
    FailureObservation,
    conditional_failure_rate,
    failure_overlap,
    joint_failure_rate,
    unique_detection_rate,
)
from stephanie.portfolio.outcome import PortfolioOutcome
from stephanie.portfolio.plan import PortfolioBudget, PortfolioPlan
from stephanie.portfolio.planner import PortfolioPlanner
from stephanie.portfolio.policy import PortfolioPolicy
from stephanie.portfolio.roles import (
    ANCHORED_ROLES,
    INDEPENDENT_ROLES,
    PortfolioRole,
)
from stephanie.portfolio.synthesis import PortfolioSynthesizer
from stephanie.portfolio.value import (
    MarginalValueComponents,
    UniqueContribution,
)
from stephanie.portfolio.verification import (
    VerificationResult,
    VerifierRegistry,
    exact_match_verifier,
    json_schema_verifier,
    numeric_check_verifier,
)

__all__ = [
    "PortfolioRole", "INDEPENDENT_ROLES", "ANCHORED_ROLES",
    "PortfolioCandidate", "PortfolioPlan", "PortfolioBudget",
    "PortfolioPolicy", "PortfolioPlanner", "PortfolioExecutor", "PortfolioExecution",
    "PortfolioEvaluator", "Disagreement", "DisagreementAnalyzer",
    "VerificationResult", "VerifierRegistry",
    "exact_match_verifier", "json_schema_verifier", "numeric_check_verifier",
    "PortfolioSynthesizer", "PortfolioOutcome",
    "ModelRolePerformance", "role_performance",
    "FailureObservation", "joint_failure_rate", "conditional_failure_rate",
    "failure_overlap", "unique_detection_rate",
    "UniqueContribution", "MarginalValueComponents",
    "PortfolioDiagnostic",
    "NO_ELIGIBLE_MODEL", "BUDGET_EXCEEDED", "ROLE_UNFILLED",
    "INDEPENDENCE_VIOLATION", "EXECUTION_FAILED", "EVALUATION_MISSING",
    "DISAGREEMENT_UNRESOLVED", "VERIFICATION_FAILED", "SYNTHESIS_FAILED",
    "CORRELATED_FAILURE", "ZERO_MARGINAL_GAIN",
    "FACTUAL", "RECOMMENDATION", "ASSUMPTION", "INTERPRETATION",
    "MISSING_INFORMATION", "METHOD", "CONFIDENCE",
]
