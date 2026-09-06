"""Fast experiment-harness tests (no models): adjudication, parsing, metrics."""
from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

from run_portfolio_experiment_001 import build_cases, parse_findings
from stephanie.portfolio.experiment import (
    DeterministicAdjudicator,
    ExperimentArm,
    Finding,
    FindingClass,
    PortfolioBenchmarkCase,
    compute_arm_metrics,
)


def _finding(category="ORPHAN_CONCEPT", claim="Examples lack any declared role.", case="c1"):
    return Finding(finding_id=f"{case}-f", case_id=case, arm="A", candidate_id="cand",
                   category=category, claim=claim)


def test_category_label_matches_code():
    cases = build_cases(["07"], 3500)
    case = next(c for c in cases if c.case_id == "argreview-07")
    expected = next(e for e in case.expected if e.code == "ORPHAN_CONCEPT")
    print(".ORPHAN_CONCEPT keywords:", expected.keywords)
    # Bare label without defect content is NOT a detection.
    adjudicated = DeterministicAdjudicator().adjudicate([_finding()], case)
    finding, _ = adjudicated[0]
    assert finding.classification == FindingClass.UNVERIFIABLE
    # Label plus defect content is.
    content_claim = "Example-roles is introduced but never used by any argument."
    adjudicated2 = DeterministicAdjudicator().adjudicate(
        [_finding(claim=content_claim)], case)
    finding2, _ = adjudicated2[0]
    assert finding2.classification == FindingClass.TRUE_POSITIVE
    assert finding2.matched_code == "ORPHAN_CONCEPT"


def test_fp_trap_outranks_tp_match():
    cases = build_cases(["07"], 3500)
    case = next(c for c in cases if c.case_id == "argreview-07")
    finding = _finding(category="ORPHAN_SECTION",
                       claim="Section on metrics is an orphan section with no linked concept.")
    result, _ = DeterministicAdjudicator().adjudicate([finding], case)[0]
    assert result.classification == FindingClass.FALSE_POSITIVE


def test_unmatched_is_unverifiable_not_fp():
    cases = build_cases(["07"], 3500)
    case = next(c for c in cases if c.case_id == "argreview-07")
    finding = _finding(category="OTHER", claim="The prose is somewhat dry in paragraph two.")
    result, _ = DeterministicAdjudicator().adjudicate([finding], case)[0]
    assert result.classification == FindingClass.UNVERIFIABLE


def test_parse_truncated_json_salvages_complete_objects():
    findings = parse_findings(
        '[{"category": "X", "claim": "abc"}, {"category": "Y", "claim": "def"',
        "c", "A", "cand")
    assert len(findings) == 1 and findings[0].category == "X"


def test_parse_valid_json_array():
    findings = parse_findings(
        '[{"category": "DUPLICATE_ID", "claim": "koan id collided", "location": "ch7"}]',
        "c", "A", "cand")
    assert len(findings) == 1 and findings[0].category == "DUPLICATE_ID"


def test_metrics_unknown_rates_stay_unknown():
    from stephanie.portfolio.experiment.run import PortfolioExperimentRun

    run = PortfolioExperimentRun(run_id="r", case_id="c", task_type="t",
                                 arm=ExperimentArm.A_PRIMARY_ONLY, repetition=0,
                                 primary_request_id="p")
    metrics = compute_arm_metrics(ExperimentArm.A_PRIMARY_ONLY, [run], {})
    assert metrics.false_positive_rate is None
    assert metrics.unique_per_million_tokens is None


def test_primary_errors_caught_counts_against_empty_baseline():
    from stephanie.portfolio.experiment.run import PortfolioExperimentRun

    from stephanie.portfolio.experiment import Finding as _F

    tp = _F(finding_id="f", case_id="c", arm="D", candidate_id="cand",
            category="ORPHAN_CONCEPT", claim="x", classification=FindingClass.TRUE_POSITIVE,
            matched_code="ORPHAN_CONCEPT")
    run = PortfolioExperimentRun(run_id="r", case_id="c", task_type="t",
                                 arm=ExperimentArm.D_BREADTH, repetition=0,
                                 primary_request_id="p", findings=(tp,))
    metrics = compute_arm_metrics(ExperimentArm.D_BREADTH, [run], {"c": set()})
    assert metrics.primary_errors_caught == 1
    assert metrics.primary_cases == 1
