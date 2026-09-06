"""Paired corpus tests (no models): twin FP alarms, labels, DD math."""
from __future__ import annotations

from stephanie.portfolio.experiment import (
    ExperimentArm,
    Finding,
    FindingClass,
    PortfolioBenchmarkCase,
    DeterministicAdjudicator,
    ExpectedFinding,
)
from stephanie.portfolio.experiment.paired import paired_analysis, verdict


def _case(case_id, defect_present, twin_of=None):
    return PortfolioBenchmarkCase.freeze(
        case_id=case_id, task_type="t", prompt="p", source_text="s",
        expected=(ExpectedFinding(code="ORPHAN_CONCEPT",
                                  keywords=("example-roles", "unused", "argument")),),
        metadata={"pair_id": "p1", "defect_present": defect_present,
                  "acceptable_labels": ["ORPHAN_CONCEPT", "MISSING_PREREQUISITE"],
                  "twin_of": twin_of},
    )


def _finding(code_match=True, category="ORPHAN_CONCEPT"):
    claim = ("The example-roles concept is introduced but never used by any argument."
             if code_match else "The prose is somewhat dry.")
    return Finding(finding_id="f", case_id="c", arm="A", candidate_id="cand",
                   category=category, claim=claim)


def test_clean_twin_match_is_false_alarm():
    twin = _case("p1-clean", False, twin_of="p1-defect")
    result, adj = DeterministicAdjudicator().adjudicate([_finding()], twin)[0]
    assert result.classification == FindingClass.FALSE_POSITIVE
    assert adj.method == "clean_twin_false_alarm"


def test_acceptable_label_tolerance():
    defect = _case("p1-defect", True)
    result, _ = DeterministicAdjudicator().adjudicate(
        [_finding(category="MISSING_PREREQUISITE")], defect)[0]
    assert result.classification == FindingClass.TRUE_POSITIVE


def test_wrong_label_with_content_still_counts_via_keywords():
    defect = _case("p1-defect", True)
    finding = Finding(finding_id="f", case_id="c", arm="A", candidate_id="cand",
                      category="OTHER",
                      claim="The example-roles concept is unused by any argument here.")
    result, _ = DeterministicAdjudicator().adjudicate([finding], defect)[0]
    assert result.classification == FindingClass.TRUE_POSITIVE


def test_paired_dd_math_and_verdict():
    defect = _case("p1-defect", True)
    twin = _case("p1-clean", False, twin_of="p1-defect")

    def rec(case_id, arm, findings):
        return {"run_id": f"{case_id}-{arm}", "case_id": case_id, "arm": arm,
                "primary_request_id": "same",
                "candidate_roles": {"cand": "primary"},
                "raw_outputs": {"cand": "x"},
                "findings": [{"candidate_id": "cand", "claim": f.claim[:50],
                              "classification": f.classification.value,
                              "matched_code": f.matched_code} for f in findings]}

    tp, _ = DeterministicAdjudicator().adjudicate([_finding()], defect)[0]
    fp, _ = DeterministicAdjudicator().adjudicate([_finding()], twin)[0]
    clean, _ = DeterministicAdjudicator().adjudicate(
        [_finding(code_match=False, category="OTHER")], twin)[0]
    assert fp.classification == FindingClass.FALSE_POSITIVE
    assert clean.classification == FindingClass.UNVERIFIABLE

    records = [
        rec("p1-defect", "A", [tp]), rec("p1-clean", "A", [clean]),
        rec("p1-defect", "E", [tp]), rec("p1-clean", "E", [clean]),
    ]
    analysis = paired_analysis(records, [defect, twin])
    assert analysis["recall"]["A"] == 1.0
    assert analysis["clean_fp"]["A"] == 0.0
    assert analysis["dd"]["A"] == 1.0
    assert analysis["identity_violations"] == []
    assert analysis["raw_missing"] == []
    status, reasons = verdict(analysis)
    assert status == "VALIDATED", reasons
