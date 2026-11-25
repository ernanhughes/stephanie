from __future__ import annotations

import json
from pathlib import Path

import pytest

from stephanie.improvement.models import Candidate
from stephanie.improvement.sc3 import (
    PYTHON_ORACLE, _compare, _score, _sha, _worker_sources, freeze, generate_hypothesis, run, validate_candidate,
)
from stephanie.improvement.models import digest


def worker_row(scenario: str, arm: str, ordinal: int, output: str | None,
               *, status: str = "AWAITING_ACCEPTANCE", invocations: int = 1,
               strategy: str | None = None) -> dict:
    task_id = f"{scenario}_{ordinal:02d}"
    strategy = strategy or ("structured.v1" if arm == "B" else "adaptive.v1")
    request_id = f"req-{scenario}-{arm}-{ordinal}"
    result = {"schema": "worker.work-result.v1", "request_id": request_id, "status": status,
              "strategy_execution": {"strategy": {"version": "1"},
                                     "resources": {"elapsed_ms": 10, "invocation_count": invocations,
                                                   "observed_output_tokens": 6}}}
    if output is not None:
        result["output"] = output
    if status in {"FAILED", "BLOCKED"}:
        result["unresolved"] = [{"code": "fixture_failure", "message": "deterministic test failure"}]
    return {"request": {"task_id": task_id, "execution": {"strategy_id": strategy},
                         "metadata": {"stage_id": f"SC3:{scenario}_{arm}_{ordinal:02d}"}},
            "response": result, "ledger_record": {"output_digest": "0" * 64}}


def cohort(scenario: str, baseline: tuple[str | None, str], candidate: tuple[str | None, str], *, n: int = 3,
           baseline_calls: int = 1, candidate_calls: int = 1,
           candidate_strategy: str = "adaptive.v1") -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        rows.append(worker_row(scenario, "B", i, baseline[0], status=baseline[1], invocations=baseline_calls))
        rows.append(worker_row(scenario, "C", i, candidate[0], status=candidate[1], invocations=candidate_calls,
                               strategy=candidate_strategy))
    return rows


def write_oracle(root: Path, protocol_digest: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "frozen-evaluator.json").write_text(json.dumps({**PYTHON_ORACLE, "protocol_digest": protocol_digest}), encoding="utf-8")


def test_praxis_controls_cover_improvement_no_benefit_regression_and_insufficient(tmp_path):
    root = tmp_path
    write_oracle(root, "a" * 64)
    c1 = _compare(cohort("C1", (None, "FAILED"), ('{"answer":"correct"}', "AWAITING_ACCEPTANCE")),
                  "C1", root, "a" * 64, "worker-head", "experiment-1")
    c2 = _compare(cohort("C2", ('{"answer":"correct"}', "AWAITING_ACCEPTANCE"),
                         ('{"answer":"correct"}', "AWAITING_ACCEPTANCE")), "C2", root, "b" * 64,
                  "worker-head", "experiment-2")
    c3 = _compare(cohort("C3", ('{"answer":"correct"}', "AWAITING_ACCEPTANCE"),
                         ('{"answer":"regressed"}', "AWAITING_ACCEPTANCE")), "C3", root, "c" * 64,
                  "worker-head", "experiment-3")
    c4 = _compare(cohort("C4", ('{"answer":"correct"}', "AWAITING_ACCEPTANCE"),
                         ('{"answer":"correct"}', "AWAITING_ACCEPTANCE"), n=1), "C4", root, "d" * 64,
                  "worker-head", "experiment-4")
    assert c1["praxis"]["decision"] == "retain"
    assert c2["praxis"]["decision"] == "reject"
    assert c3["praxis"]["decision"] == "reject"
    assert c4["praxis"]["decision"] == "insufficient_evidence"


def test_additional_compute_is_visible_and_withheld_from_positive_recommendation(tmp_path):
    write_oracle(tmp_path, "e" * 64)
    result = _compare(cohort("C5", ('{"answer":"regressed"}', "AWAITING_ACCEPTANCE"),
                             ('{"answer":"correct"}', "AWAITING_ACCEPTANCE"), candidate_calls=2,
                             candidate_strategy="corrective.v1"), "C5", tmp_path, "e" * 64,
                     "worker-head", "experiment-5")
    assert result["praxis"]["decision"] == "retain"
    assert result["resource_comparison"]["additional_compute"] is True
    assert result["recommendation"] == "WITHHOLD_RESOURCE_CONFOUNDED"
    assert result["resource_comparison"]["cost_status"] == "unknown"


def test_execution_failure_remains_failed_and_evaluable_as_false():
    row = worker_row("C6_FAIL", "B", 1, None, status="FAILED")
    assert _score(row, PYTHON_ORACLE) == (0.0, False)


def test_missing_frozen_evaluator_fails_closed(tmp_path):
    rows = cohort("C1", (None, "FAILED"), ('{"answer":"correct"}', "AWAITING_ACCEPTANCE"))
    with pytest.raises(FileNotFoundError, match="frozen-evaluator.json"):
        _compare(rows, "C1", tmp_path, "f" * 64, "worker-head", "experiment-no-evaluator")


def test_candidate_mutation_is_rejected_before_any_worker_execution(tmp_path):
    source_digest = _sha(json.dumps(_worker_sources(), sort_keys=True).encode())
    evidence_id = "worker-strategy-source-" + source_digest[:12]
    hypothesis = {"statement": "JSON fence handling can prevent a representation-only validation failure.",
                  "rationale": "The inspected Worker strategy applies a deterministic fence transform before schema validation.",
                  "uncertainty": "The deterministic fixture does not establish model quality or operational effectiveness.",
                  "evidence_ids": [evidence_id]}
    artifact = {"hypothesis": hypothesis, "hypothesis_digest": digest(hypothesis), "evidence_id": evidence_id}
    (tmp_path / "ai-hypothesis.json").write_text(json.dumps(artifact), encoding="utf-8")
    freeze(tmp_path)
    protocol = json.loads((tmp_path / "frozen-protocol.json").read_text(encoding="utf-8"))
    candidate = Candidate.model_validate(json.loads((tmp_path / "candidate.json").read_text(encoding="utf-8")))
    invalid = candidate.model_copy(update={"parameters": {**candidate.parameters, "strategy_id": "unrestricted.shell.v1"}})
    with pytest.raises(ValueError, match="candidate changed after freeze"):
        validate_candidate(tmp_path, protocol, invalid)


def test_shared_ai_hypothesis_claim_prevents_retry_before_dispatch(tmp_path, monkeypatch):
    from stephanie.improvement.engine import ImprovementEngine
    claim = tmp_path / "ai" / "sc3-hypothesis-invocation.claim"
    claim.parent.mkdir(parents=True)
    claim.write_text('{"status":"invocation_may_be_dispatched"}\n', encoding="utf-8")

    def forbidden_call(*_args, **_kwargs):
        raise AssertionError("provider call must not happen after a prior invocation claim")

    monkeypatch.setattr(ImprovementEngine, "investigate_with_ai", forbidden_call)
    with pytest.raises(RuntimeError, match="already claimed"):
        generate_hypothesis(tmp_path)


def test_sc3_run_claim_prevents_duplicate_worker_dispatch(tmp_path, monkeypatch):
    evidence_digest = _sha(json.dumps(_worker_sources(), sort_keys=True).encode())
    evidence_id = "worker-strategy-source-" + evidence_digest[:12]
    hypothesis = {"statement": "JSON fence handling can prevent a representation-only validation failure.",
                  "rationale": "The inspected Worker strategy applies a deterministic fence transform before schema validation.",
                  "uncertainty": "The deterministic fixture does not establish model quality or operational effectiveness.",
                  "evidence_ids": [evidence_id]}
    (tmp_path / "ai-hypothesis.json").write_text(json.dumps({
        "hypothesis": hypothesis, "hypothesis_digest": digest(hypothesis),
        "evidence_id": evidence_id}), encoding="utf-8")
    freeze(tmp_path)
    claim = tmp_path / "sc3-run.claim"
    claim.write_text(json.dumps({"status": "dispatch_may_have_started"}), encoding="utf-8")

    def forbidden_dispatch(*_args, **_kwargs):
        raise AssertionError("Worker dispatch must not repeat after a prior claim")

    monkeypatch.setattr("stephanie.improvement.sc3.subprocess.run", forbidden_dispatch)
    with pytest.raises(RuntimeError, match="already claimed"):
        run(tmp_path)
