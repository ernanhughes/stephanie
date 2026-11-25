"""SC3 bounded Worker/Praxis integration runner for existing Worker strategy fixtures."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from praxis.effectiveness import (
    AcceptanceResult, EffectivenessRecord, ExecutionRef, PromotionPolicy,
    compare_effectiveness,
)

from .engine import ImprovementEngine
from .integrations import compare_with_praxis
from .models import Candidate, Hypothesis, digest

SC3_ROOT = Path(__file__).resolve().parents[2]
WORKER_ROOT = Path("C:/Projects/worker")
PYTHON_ORACLE = {"schema_version": "sc3.independent-oracle.v1", "answer": "correct", "evaluator": "exact-json-answer.v1"}
WORKER_FIXTURE_RESPONSE = '```json\n{"answer":"correct"}\n```'
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _worker_sources() -> dict[str, str]:
    names = ["src/core/worker.ts", "src/server/server.ts", "src/executors/model.ts",
             "src/strategies/catalog.ts", "src/strategies/runtime.ts",
             "src/core/execution-ledger.ts", "src/core/execution-summary.ts"]
    return {name: _sha((WORKER_ROOT / name).read_bytes()) for name in names}


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _worker_request(task_id: str, task_marker: str, experiment_id: str, strategy: str, *, output_tokens: int = 64) -> dict:
    execution: dict[str, str] = {"strategy_id": strategy, "model": "fixture_model", "policy": "sc3.local"}
    if strategy == "adaptive.v1":
        execution["normalization_policy"] = "json-fence.v1"
    return {
        "schema": "worker.work-request.v1", "app": "stephanie-sc3", "task_id": task_id,
        "objective": f"Return one JSON object matching the supplied public structure. [[SC3_TASK:{task_marker}]]",
        "execution": execution,
        "parameters": {"data_policy": "local_only", "timeout_ms": 5_000,
                       "max_output_tokens": output_tokens, "output_schema": SCHEMA},
        "options": {"acceptance": "human"},
        "metadata": {"operation_id": experiment_id, "stage_id": f"SC3:{task_marker}"},
    }


def _build_requests(experiment_id: str) -> tuple[list[dict], dict[str, list[str]]]:
    requests: list[dict] = []
    responses: dict[str, list[str]] = {}

    def add(scenario: str, arm: str, n: int, strategy: str, outputs: list[str]) -> None:
        task_marker = f"{scenario}_{arm}_{n:02d}"
        task_id = f"{scenario}_{n:02d}" if scenario != "C6_FAIL" else task_marker
        requests.append(_worker_request(task_id, task_marker, experiment_id, strategy,
                                        output_tokens=128))
        responses[task_marker] = outputs

    for n in range(1, 4):
        add("C1", "B", n, "structured.v1", [WORKER_FIXTURE_RESPONSE])
        add("C1", "C", n, "adaptive.v1", [WORKER_FIXTURE_RESPONSE])
        add("C2", "B", n, "structured.v1", ['{"answer":"correct"}'])
        add("C2", "C", n, "adaptive.v1", ['{"answer":"correct"}'])
        add("C3", "B", n, "structured.v1", ['{"answer":"correct"}'])
        add("C3", "C", n, "adaptive.v1", ['{"answer":"regressed"}'])
        add("C5", "B", n, "structured.v1", ['{"answer":"regressed"}'])
        add("C5", "C", n, "corrective.v1", ['{}', '{"answer":"correct"}'])
    add("C4", "B", 1, "structured.v1", ['{"answer":"correct"}'])
    add("C4", "C", 1, "adaptive.v1", ['{"answer":"correct"}'])
    add("C6_FAIL", "B", 1, "structured.v1", [])
    return requests, responses


def freeze(root: str | Path, *, mission_observation: dict | None = None) -> dict:
    """Freeze source identities, immutable arms, task prompts and separate oracle."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    hypothesis_path = root / "ai-hypothesis.json"
    if not hypothesis_path.exists():
        raise FileNotFoundError("run the once-only bounded sc3-hypothesis step before freezing the candidate")
    ai_hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    if ai_hypothesis.get("hypothesis_digest") != digest(ai_hypothesis.get("hypothesis")):
        raise ValueError("shared-AI hypothesis digest mismatch")
    if not ai_hypothesis.get("hypothesis", {}).get("evidence_ids"):
        raise ValueError("shared-AI hypothesis must cite the supplied sanitized Worker evidence")
    source_files = _worker_sources()
    worker_head = _git_head(WORKER_ROOT)
    protocol_seed = {"task_family": "Worker structured-output handling / unambiguous JSON-fence normalization",
                     "worker_head": worker_head, "worker_source_sha256": source_files,
                     "baseline": {"strategy_id": "structured.v1", "version": "1"},
                     "candidate": {"strategy_id": "adaptive.v1", "version": "1", "normalization_policy": "json-fence.v1"},
                     "hypothesis_digest": ai_hypothesis["hypothesis_digest"],
                     "hypothesis_evidence_id": ai_hypothesis["evidence_id"],
                     "resource_limits": {"per_execution_timeout_ms": 5000, "max_output_tokens": 128,
                                          "max_strategy_invocations": 2, "total_model_invocations": 0,
                                          "provider_cost_usd": 0, "data_policy": "local_only"},
                     "worker_service": "isolated WorkerService + createWorkerServer HTTP endpoint, 127.0.0.1 ephemeral port",
                     "execution_ledger": "<experiment-root>/worker/worker-execution-ledger.jsonl",
                     "other_worker_ledgers": "disabled", "mission_write": False}
    if mission_observation is not None:
        if (not isinstance(mission_observation, dict)
                or mission_observation.get("source_identity", "").startswith("mission:episode/") is False
                or not mission_observation.get("source_digest")
                or not mission_observation.get("evidence_id")):
            raise ValueError("Mission observation identity and digest are required for this protocol")
        protocol_seed["mission_observation"] = mission_observation
    experiment_id = _sha(json.dumps(protocol_seed, sort_keys=True).encode())[:24]
    requests, provider_responses = _build_requests(experiment_id)
    protocol = {**protocol_seed, "experiment_id": experiment_id,
                "requests": requests, "provider_responses": provider_responses}
    # Expected answer remains in a separate artifact, never in Worker input.
    evaluator_digest = digest(PYTHON_ORACLE)
    # Reuse Stephanie's immutable Candidate/Hypothesis representations.
    evidence_digest = _sha(json.dumps(source_files, sort_keys=True).encode())
    evidence_id = "worker-strategy-source-" + evidence_digest[:12]
    engine = ImprovementEngine(root)
    evidence = {"evidence_id": evidence_id, "source_system": "worker", "source_identity": f"git:{worker_head}:worker-strategy-runtime",
                "source_digest": evidence_digest, "observed_facts": ("Worker exposes versioned structured.v1 and adaptive.v1 strategies.",),
                "limitations": ("Source worktree contains concurrent modifications; per-file content digests are frozen in SC3 protocol.",),
                "evaluation_status": "UNKNOWN", "acceptance_status": "UNKNOWN", "cost_status": "unknown"}
    engine.import_evidence([evidence])
    if ai_hypothesis["evidence_id"] != evidence_id:
        raise ValueError("shared-AI hypothesis was generated from a different Worker source identity")
    hypothesis = Hypothesis(**ai_hypothesis["hypothesis"])
    candidate_payload = {"experiment_id": experiment_id, "strategy_id": "adaptive.v1", "normalization_policy": "json-fence.v1",
                         "resource_limits": protocol_seed["resource_limits"], "evaluator_digest": evaluator_digest}
    candidate_id = _sha(json.dumps(candidate_payload, sort_keys=True).encode())[:24]
    mission_evidence_id = (mission_observation or {}).get("evidence_id")
    evidence_ids = tuple(sorted(set(hypothesis.evidence_ids) | {evidence_id}
                                | ({mission_evidence_id} if mission_evidence_id else set())))
    evidence_rows = {row["evidence_id"]: row for row in engine._read(engine.evidence_path, [])}
    evidence_digests = tuple(sorted((ident, digest(evidence_rows[ident])) for ident in evidence_ids
                                    if ident in evidence_rows))
    if len(evidence_digests) != len(evidence_ids):
        raise ValueError("frozen candidate references evidence absent from Stephanie's evidence store")
    candidate = Candidate(candidate_id=candidate_id, name="adaptive JSON-fence normalization",
                          intervention="Run Worker adaptive.v1 with its explicit json-fence.v1 normalization policy before structural validation.",
                          parameters={"strategy_id": "adaptive.v1", "normalization_policy": "json-fence.v1", "max_model_invocations": 2,
                                      "timeout_ms": 5000, "max_output_tokens": 128, "data_policy": "local_only"},
                         evidence_ids=evidence_ids, evidence_digests=evidence_digests, hypothesis=hypothesis,
                          protected_evaluator_digest=evaluator_digest, requires_additional_compute=False)
    candidate_json = candidate.model_dump(mode="json")
    protocol["candidate_digest"] = digest(candidate_json)
    baseline = {"baseline_id": "worker-structured-v1", "strategy_id": "structured.v1", "version": "1",
                "digest": digest(protocol_seed["baseline"]), "immutable": True}
    protocol["baseline_digest"] = digest(baseline)
    protocol["evaluator_digest"] = evaluator_digest
    protocol_digest = digest(protocol)
    protocol["protocol_digest"] = protocol_digest
    _json(root / "frozen-protocol.json", protocol)
    _json(root / "frozen-evaluator.json", {**PYTHON_ORACLE, "protocol_digest": protocol_digest,
                                             "evaluator_digest": evaluator_digest})
    _json(root / "candidate.json", candidate_json)
    _json(root / "baseline.json", baseline)
    return {"experiment_id": experiment_id, "protocol_digest": protocol_digest, "candidate_id": candidate_id,
            "candidate_digest": protocol["candidate_digest"], "evaluator_digest": evaluator_digest,
            "mission_observation": protocol_seed.get("mission_observation"),
            "requests": len(requests), "output": str(root / "frozen-protocol.json")}


def generate_hypothesis(root: str | Path, *, model_name: str = "qwen2.5:0.5b") -> dict:
    """Make exactly one shared-AI hypothesis call from sanitized, evaluator-free Worker facts."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = root / "ai-hypothesis.json"
    ai_root = root / "ai"
    ai_root.mkdir(parents=True, exist_ok=True)
    claim_path = ai_root / "sc3-hypothesis-invocation.claim"
    if out.exists() or (ai_root / "ai-executions.json").exists():
        raise RuntimeError("SC3 hypothesis step is once-only; existing artifacts prohibit retry")
    try:
        descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("SC3 hypothesis invocation was already claimed; retry is prohibited") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as claim:
        claim.write(json.dumps({"status": "invocation_may_be_dispatched", "created_at": datetime.now(timezone.utc).isoformat()}) + "\n")
        claim.flush()
        os.fsync(claim.fileno())
    source_files = _worker_sources()
    worker_head = _git_head(WORKER_ROOT)
    evidence_digest = _sha(json.dumps(source_files, sort_keys=True).encode())
    evidence_id = "worker-strategy-source-" + evidence_digest[:12]
    from .engine import ImprovementEngine
    from .models import Evidence
    engine = ImprovementEngine(ai_root)
    evidence = Evidence(evidence_id=evidence_id, source_system="synthetic",
        source_identity=f"git:{worker_head}:worker-strategy-runtime", source_digest=evidence_digest,
        observed_facts=("Worker source defines structured.v1 schema validation and adaptive.v1 with json-fence.v1 normalization.",),
        limitations=("This is sanitized mechanism evidence for a deterministic fixture; it does not establish semantic quality or effectiveness.",),
        evaluation_status="UNKNOWN", acceptance_status="UNKNOWN", cost_status="unknown")
    engine.import_evidence([evidence.model_dump(mode="json")])
    record = engine.investigate_with_ai("ollama", model_name)
    if not record.get("qualification_eligible") or not record.get("hypothesis"):
        raise RuntimeError("shared AI did not return a qualified structured hypothesis; preserved evidence is available and retry is prohibited")
    hypothesis = record["hypothesis"]
    safe_hypothesis = {"statement": hypothesis["statement"], "rationale": hypothesis["rationale"],
                       "uncertainty": hypothesis["uncertainty"], "evidence_ids": hypothesis["evidence_ids"]}
    payload = {"schema_version": "stephanie.sc3-ai-hypothesis.v1", "evidence_id": evidence_id,
               "worker_head": worker_head, "worker_source_sha256": source_files,
               "model": record["effective_configuration"]["model"], "request_id": record["request_id"],
               "hypothesis": safe_hypothesis, "hypothesis_digest": digest(safe_hypothesis),
               "authority": "hypothesis_only", "protected_evaluator_in_context": False,
               "raw_response_artifact": str(ai_root / "ai-executions.json")}
    _json(out, payload)
    return {"status": "complete", "hypothesis_digest": payload["hypothesis_digest"],
            "model": payload["model"], "actual_model_invocations": 1, "output": str(out),
            "raw_response_in_default_output": False}


def _record(row: dict, score: float, goal: bool | None, *, protocol_digest: str,
            worker_head: str, experiment_id: str) -> EffectivenessRecord:
    request, result, ledger = row["request"], row["response"], row.get("ledger_record") or {}
    strategy = request["execution"]["strategy_id"]
    task = request["task_id"]
    strat = result.get("strategy_execution") or {}
    resources = strat.get("resources") or {}
    criterion = AcceptanceResult(criterion_id="frozen-json-answer", statement="Output exactly matches the frozen independent JSON oracle.",
                                 source="sc3-independent-evaluator", passed=goal,
                                 evaluator_id="sc3-json-answer-v1", evidence=["output digest recorded; expected value not included"])
    marker = str((request.get("metadata") or {}).get("stage_id") or "")
    execution = ExecutionRef(execution_id=str(result.get("request_id") or task), proposal_key="unsubmitted-local-experiment",
                             correlation_id=experiment_id, application_id="stephanie-sc3", action_id=strategy,
                             status=str(result.get("status") or "UNKNOWN"), risk="isolated_fixture", approval_required=False,
                             duration_seconds=(resources.get("elapsed_ms") / 1000) if isinstance(resources.get("elapsed_ms"), (int, float)) else None,
                             result_schema=str(result.get("schema") or "worker.work-result.v1"), artifact_present=bool(ledger),
                             error=(result.get("unresolved") or [{}])[0].get("code") if result.get("unresolved") else None)
    record = EffectivenessRecord(record_id=f"{task}:{result.get('request_id','missing')}", goal_id=task,
        goal_title="Worker JSON-fence normalization controlled fixture", task_family="worker.structured-output",
        strategy_id=strategy, strategy_version=str((strat.get("strategy") or {}).get("version") or "1"),
        capability_versions={"worker_api": "worker.work-request.v1", "strategy_catalogue": "worker.execution-strategy.v1"},
        executions=[execution], acceptance_criteria=[criterion], acceptance_provenance="independent_fixture_evaluator_not_human_acceptance",
        execution_success=bool(result.get("status") in {"AWAITING_ACCEPTANCE", "ACCEPTED"}), goal_achievement=goal,
        artifact_quality=score, duration_seconds=execution.duration_seconds, cost_usd=None,
        failure=None if execution_success_status(result) else str((result.get("unresolved") or [{}])[0].get("code", "execution_failed")),
        retries=max(0, int(resources.get("invocation_count", 0) or 0) - 1), baseline_comparison=None,
        decision="insufficient_evidence", rationale="Per-execution evidence; Praxis comparison is computed over frozen arm cohorts.",
        eck_experience_refs=[], independent_evaluation={"evaluator": "sc3-json-answer-v1", "evaluator_digest": digest(PYTHON_ORACLE),
            "output_digest": digest(result.get("output")), "status": "scored" if goal is not None else "inconclusive"},
        provenance={"experiment_id": experiment_id, "protocol_digest": protocol_digest, "worker_source_commit": worker_head,
                    "worker_request_id": result.get("request_id"), "task_id": task,
                    "worker_invocation_count": int(resources.get("invocation_count", 0) or 0),
                    "worker_ledger_digest": ledger.get("output_digest") or ledger.get("strategy_execution", {}).get("normalized_artifact", {}).get("output_digest"),
                    "resource_observation_source": "Worker strategy_execution.resources; deterministic provider usage is synthetic",
                    "cost_status": "unknown", "acceptance_status": "UNKNOWN", "formal_acceptance": False},
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        arm="baseline" if "_B_" in marker or "BASELINE" in marker else "candidate", pair={"pair_id": task, "task_id": task})
    return record.require_valid()


def execution_success_status(result: dict) -> bool:
    return result.get("status") in {"AWAITING_ACCEPTANCE", "ACCEPTED"}


def _score(row: dict, oracle: dict) -> tuple[float, bool | None]:
    result = row["response"]
    output = result.get("output")
    if not isinstance(output, str):
        return 0.0, False if result.get("status") in {"FAILED", "BLOCKED"} else None
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError):
        return 0.0, False
    correct = isinstance(parsed, dict) and parsed.get("answer") == oracle["answer"]
    return (1.0 if correct else 0.0), correct


def _compare(rows: list[dict], scenario: str, root: Path, protocol_digest: str, worker_head: str, experiment_id: str) -> dict:
    oracle = json.loads((root / "frozen-evaluator.json").read_text(encoding="utf-8"))
    selected = [row for row in rows if row["request"]["task_id"].startswith(scenario + "_")]
    records: list[EffectivenessRecord] = []
    by_arm: dict[str, list[EffectivenessRecord]] = {"baseline": [], "candidate": []}
    execution_evidence: list[dict] = []
    for row in selected:
        score, goal = _score(row, oracle)
        item = _record(row, score, goal, protocol_digest=protocol_digest, worker_head=worker_head, experiment_id=experiment_id)
        records.append(item); by_arm[item.arm or "baseline"].append(item)
        execution_evidence.append({"task_id": item.goal_id, "request_id": row["response"].get("request_id"),
            "strategy": item.strategy_id, "result_status": row["response"].get("status"), "score": score,
            "goal_achievement": goal, "invocations": (row["response"].get("strategy_execution") or {}).get("resources", {}).get("invocation_count"),
            "output_tokens": (row["response"].get("strategy_execution") or {}).get("resources", {}).get("observed_output_tokens"),
            "cost_status": "unknown", "worker_ledger_recorded": bool(row.get("ledger_record"))})
    comparison = compare_with_praxis(by_arm["baseline"], by_arm["candidate"], praxis_module=type("Praxis", (), {"compare_effectiveness": compare_effectiveness}))
    # Match actual strategy resources by authoritative Worker request ID;
    # paired arms deliberately share task_id.
    b_invocations = sum(next((e.get("invocations") or 0 for e in execution_evidence
                              if e["request_id"] == r.provenance.get("worker_request_id")), 0)
                        for r in by_arm["baseline"])
    c_invocations = sum(next((e.get("invocations") or 0 for e in execution_evidence
                              if e["request_id"] == r.provenance.get("worker_request_id")), 0)
                        for r in by_arm["candidate"])
    compute_confounded = c_invocations > b_invocations
    answer = {"scenario": scenario, "praxis": comparison["comparison"], "praxis_callable": comparison["mode"],
              "operational_evaluation": False, "records": [r.to_dict() for r in records],
              "execution_evidence": execution_evidence,
              "resource_comparison": {"baseline_invocations": b_invocations, "candidate_invocations": c_invocations,
                                      "additional_compute": compute_confounded, "cost_usd": None, "cost_status": "unknown",
                                      "note": "Praxis effectiveness comparer does not consume invocation/cost dimensions."},
              "recommendation": "WITHHOLD_RESOURCE_CONFOUNDED" if compute_confounded else comparison["comparison"]["decision"]}
    _json(root / f"praxis-{scenario.lower()}.json", answer)
    return answer


def run(root: str | Path) -> dict:
    root = Path(root).resolve()
    if (root / "sc3-run.claim").exists():
        raise RuntimeError("SC3 run already claimed; inspect authoritative Worker ledger and artifacts; retry is prohibited")
    protocol_path = root / "frozen-protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError("freeze SC3 protocol before running")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_digest"] != digest({k: v for k, v in protocol.items() if k != "protocol_digest"}):
        raise ValueError("frozen protocol digest mismatch")
    if _worker_sources() != protocol["worker_source_sha256"] or _git_head(WORKER_ROOT) != protocol["worker_head"]:
        raise ValueError("Worker source changed after protocol freeze; re-freeze before execution")
    candidate = validate_candidate(root, protocol)
    runner = Path(__file__).with_name("sc3_worker_runner.ts")
    bun = shutil.which("bun")
    if not bun:
        raise RuntimeError("Bun is required to run the isolated actual Worker HTTP integration")
    # The protocol root is the execution idempotency boundary. Claim it
    # exclusively before any Worker dispatch; an interrupted/unknown run must
    # be reconciled from its Worker ledger and artifacts, never retried here.
    claim_path = root / "sc3-run.claim"
    if claim_path.exists():
        raise RuntimeError("SC3 run already claimed; inspect authoritative Worker ledger and artifacts; retry is prohibited")
    if any(path.exists() for path in (root / "worker-results.json",
                                      root / "integrated-results.json",
                                      root / "worker" / "worker-execution-ledger.jsonl")):
        raise RuntimeError("SC3 execution artifacts already exist; reconcile the existing run; retry is prohibited")
    claim = {"schema_version": "stephanie.sc3-run-claim.v1",
             "experiment_id": protocol["experiment_id"],
             "protocol_digest": protocol["protocol_digest"],
             "status": "dispatch_may_have_started",
             "created_at": datetime.now(timezone.utc).isoformat()}
    try:
        descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("SC3 run already claimed; inspect authoritative Worker ledger and artifacts; retry is prohibited") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as claim_file:
        claim_file.write(json.dumps(claim, sort_keys=True) + "\n")
        claim_file.flush()
        os.fsync(claim_file.fileno())
    worker_root = root / "worker"
    out_path = root / "worker-results.json"
    actual_protocol = {**protocol, "artifact_root": str(worker_root)}
    _json(root / "worker-input.json", actual_protocol)
    run_result = subprocess.run([bun, str(runner), str(root / "worker-input.json"), str(out_path)],
                                cwd=str(WORKER_ROOT), capture_output=True, text=True, timeout=120)
    if run_result.returncode:
        raise RuntimeError(f"isolated Worker runner failed: {run_result.stderr[-2000:]}")
    worker = json.loads(out_path.read_text(encoding="utf-8"))
    rows = worker["results"]
    worker_head = protocol["worker_head"]
    comparisons = [_compare(rows, scenario, root, protocol["protocol_digest"], worker_head, protocol["experiment_id"]) for scenario in ("C1", "C2", "C3", "C4", "C5")]
    c6 = next(row for row in rows if row["request"]["task_id"].startswith("C6_FAIL"))
    c6_record = {"scenario": "C6", "result_status": c6["response"].get("status"),
                 "unresolved": c6["response"].get("unresolved"), "worker_ledger_recorded": bool(c6.get("ledger_record")),
                 "inference_executed": ((c6.get("ledger_record") or {}).get("observed") or {}).get("inference_executed", "UNKNOWN"),
                 "resource_observation": (c6["response"].get("strategy_execution") or {}).get("resources")}
    # C7 is rejected locally before reaching the Worker runner.
    invalid = candidate.model_copy(update={"parameters": {**candidate.parameters, "strategy_id": "unrestricted.shell.v1"}})
    try:
        validate_candidate(root, protocol, invalid)
        c7_rejected = False
    except ValueError:
        c7_rejected = True
    _json(root / "worker-failure-c6.json", c6_record)
    _json(root / "candidate-rejection-c7.json", {"rejected_before_execution": c7_rejected,
        "requested_strategy": invalid.parameters.get("strategy_id"), "worker_request_emitted": False,
        "reason": "strategy is not in frozen candidate allowlist"})
    result = {"schema_version": "stephanie.sc3-experiment-result.v1", "experiment_id": protocol["experiment_id"],
              "protocol_digest": protocol["protocol_digest"], "candidate_id": candidate.candidate_id,
              "worker_ledger_sha256": _sha((root / "worker" / "worker-execution-ledger.jsonl").read_bytes()),
              "worker_ledger_record_count": (worker.get("storage", {}).get("ledger_read") or {}).get("record_count"),
              "worker": {k: worker[k] for k in ("worker_source_commit", "worker_service", "fixture_provider", "storage")},
              "praxis_comparisons": comparisons, "C6_execution_failure": c6_record,
              "C7_invalid_candidate": {"rejected_before_execution": c7_rejected},
              "actual_model_invocations": 0, "formal_acceptance": "UNKNOWN", "mission_writes": 0,
              "qualification": "deterministic integration using actual Worker HTTP/service and Praxis callable; scripted provider; no genuine model inference"}
    _json(root / "integrated-results.json", result)
    # Retain in the existing Stephanie CaseBook and ExperimentStore in an isolated SQLite DB.
    _retain(root, result, protocol, candidate)
    claim.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                  "integrated_results_sha256": _sha((root / "integrated-results.json").read_bytes())})
    _json(claim_path, claim)
    return {"status": "complete", "experiment_id": protocol["experiment_id"],
            "worker_requests": len(rows), "praxis_scenarios": {x["scenario"]: x["praxis"]["decision"] for x in comparisons},
            "c6_status": c6_record["result_status"], "c7_rejected": c7_rejected, "result_path": str(root / "integrated-results.json")}


def validate_candidate(root: str | Path, protocol: dict, candidate: Candidate | None = None) -> Candidate:
    root = Path(root).resolve()
    candidate = candidate or Candidate.model_validate(json.loads((root / "candidate.json").read_text(encoding="utf-8")))
    if digest(candidate.model_dump(mode="json")) != protocol["candidate_digest"]:
        raise ValueError("candidate changed after freeze")
    baseline = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
    if digest(baseline) != protocol["baseline_digest"]:
        raise ValueError("baseline changed after freeze")
    frozen_evaluator = json.loads((root / "frozen-evaluator.json").read_text(encoding="utf-8"))
    if frozen_evaluator.get("protocol_digest") != protocol["protocol_digest"] or digest({k: frozen_evaluator[k] for k in PYTHON_ORACLE}) != protocol["evaluator_digest"]:
        raise ValueError("frozen evaluator changed after freeze")
    if candidate.protected_evaluator_digest != protocol["evaluator_digest"]:
        raise ValueError("candidate does not reference the frozen evaluator")
    if candidate.parameters.get("strategy_id") != "adaptive.v1" or candidate.parameters.get("normalization_policy") != "json-fence.v1":
        raise ValueError("candidate failed pre-execution authority/compatibility validation")
    return candidate


def _retain(root: Path, result: dict, protocol: dict, candidate: Candidate) -> None:
    engine = ImprovementEngine(root)
    path = root / "experiences.json"
    experiences = engine._read(path, [])
    case = {"experiment_id": protocol["experiment_id"], "candidate_id": candidate.candidate_id,
            "original_observation": "Existing Worker structured.v1 rejects a complete unambiguous JSON fence while adaptive.v1 may normalize it.",
            "hypothesis": candidate.hypothesis.model_dump(mode="json"),
            "baseline": json.loads((root / "baseline.json").read_text(encoding="utf-8")),
            "candidate": candidate.model_dump(mode="json"),
            "frozen_protocol_digest": protocol["protocol_digest"],
            "outcome": result["praxis_comparisons"][0]["praxis"]["decision"],
            "recommendation": result["praxis_comparisons"][0]["recommendation"],
            "limitations": result["qualification"], "worker_execution_refs": [
                row["response"].get("request_id") for row in json.loads((root / "worker-results.json").read_text())["results"]],
            "praxis_refs": [f"praxis-{x['scenario'].lower()}.json" for x in result["praxis_comparisons"]],
            "independent_evaluation_refs": ["praxis-c1.json", "praxis-c2.json", "praxis-c3.json", "praxis-c4.json", "praxis-c5.json"],
            "decision_status": "NOT_PROMOTED", "reuse_eligible": False, "formal_acceptance": "UNKNOWN"}
    if not any(row.get("experiment_id") == protocol["experiment_id"] for row in experiences):
        experiences.append(case); _json(path, experiences)
    memory_engine = _local_memory_engine(root)
    book = memory_engine.casebook_store
    casebook = book.get_by_name("stephanie-sc3-integrated-experiments")
    if casebook is None:
        book.create_casebook("stephanie-sc3-integrated-experiments", description="SC3 isolated Worker/Praxis experiment references", tags=["sc3", "integration"])
    book_id = book.get_id_by_name("stephanie-sc3-integrated-experiments")
    if book_id is None:
        raise RuntimeError("SC3 CaseBook creation did not persist")
    summaries = book.list_case_summaries(book_id)
    cases_to_keep = [(protocol["experiment_id"], "Worker structured output normalization experiment", case)]
    for scenario in ("C2", "C3", "C4"):
        comparison = next(x for x in result["praxis_comparisons"] if x["scenario"] == scenario)
        failed = {"experiment_id": protocol["experiment_id"], "candidate_id": candidate.candidate_id,
                  "scenario": scenario, "outcome": comparison["praxis"]["decision"],
                  "limitations": comparison["praxis"]["rationale"], "reuse_eligible": False,
                  "promotion_status": "NOT_PROMOTED"}
        cases_to_keep.append((f"{protocol['experiment_id']}-{scenario}", f"Worker normalization {scenario} control", failed))
    for name, description, item in cases_to_keep:
        if not any(row.get("name") == name for row in summaries):
            book.add_case(book_id, None, "stephanie-integrated-experiment", name=name,
                          description=description, prompt_text=description,
                      meta={"integrated_experiment": item, "experience": item,
                            "reuse_eligible": False, "promotion_status": "NOT_PROMOTED"})
    # Existing ExperimentStore records each real Worker result in the populated C1 cohort.
    store = memory_engine.experiment_store
    experiment_name = f"sc3-worker-normalization-{protocol['experiment_id']}"
    store.get_or_create_experiment(experiment_name, domain="worker-structured-output",
        config={"protocol_digest": protocol["protocol_digest"], "candidate_id": candidate.candidate_id,
                "task_family": "worker.structured-output", "offline_fixture_provider": True})
    experiment = store.get_by_name(experiment_name, domain="worker-structured-output")
    if experiment is None:
        raise RuntimeError("SC3 ExperimentStore record was not created")
    experiment_id = int(experiment["id"])
    store.upsert_variant(experiment_id, "baseline", is_control=True,
                         payload={"strategy_id": "structured.v1", "protocol_digest": protocol["protocol_digest"]})
    store.upsert_variant(experiment_id, "candidate",
                         payload={"strategy_id": "adaptive.v1", "candidate_id": candidate.candidate_id,
                                  "protocol_digest": protocol["protocol_digest"]})
    variants = {row["name"]: row for row in store.list_variants(experiment_id)}
    c1_records = next(x for x in result["praxis_comparisons"] if x["scenario"] == "C1")["records"]
    for arm in ("baseline", "candidate"):
        for case_number, record in enumerate((row for row in c1_records if row["arm"] == arm), 1):
            metrics = {"goal_achievement": record["goal_achievement"], "artifact_quality": record["artifact_quality"],
                       "worker_invocation_count": record["provenance"].get("worker_invocation_count")}
            store.complete_trial(variant_id=int(variants[arm]["id"]), case_id=case_number,
                performance=record["artifact_quality"], metrics=metrics, cost=None,
                meta={"protocol_digest": protocol["protocol_digest"], "result_id": record["record_id"],
                      "evaluation": "independent frozen JSON oracle", "cost_status": "unknown"},
                experiment_group=f"{arm}:{case_number}")
    # Subsequent operation: retrieve from persistent CaseBook and local reference store.
    retrieved = next((row for row in book.list_case_summaries(book_id) if row.get("name") == protocol["experiment_id"]), None)
    if retrieved is None:
        raise RuntimeError("SC3 integrated experiment could not be retrieved from CaseBook")
    _json(root / "retained-experience-retrieval.json", {"casebook_name": "stephanie-sc3-integrated-experiments",
        "casebook_id": book_id, "retrieved_name": retrieved.get("name"), "retrieved": True,
        "retrieved_prior_unsuccessful_candidate": any(row.get("name") == f"{protocol['experiment_id']}-C3" for row in book.list_case_summaries(book_id)),
        "reuse_eligible": False, "successful_improvement_claim": False})
    prior = retrieve_experience(root, "worker normalization regression rejected")
    _json(root / "prior-unsuccessful-experience-retrieval.json", {"query": "worker normalization regression rejected",
        "retrieved": bool(prior), "records": prior,
        "interpretation": "retrieval demonstrates access to a prior unsuccessful control, not improved outcomes caused by retrieval"})


def _local_memory_engine(root: Path) -> ImprovementEngine:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import configure_mappers, sessionmaker
    from stephanie.memory.memory_tool import MemoryTool as _register
    from stephanie.memory.casebook_store import CaseBookStore
    from stephanie.memory.experiment_store import ExperimentStore
    from stephanie.orm.base import Base
    from stephanie.orm.casebook import CaseBookORM, CaseORM, CaseScorableORM
    from stephanie.orm.experiment import ExperimentORM, VariantORM, TrialORM, TrialMetricORM
    configure_mappers()
    from .storage import memory_db_path
    db_path = memory_db_path(root, local_name="stephanie-sc3-isolated.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = create_engine(f"sqlite:///{db_path.resolve()}")
    tables = [CaseBookORM.__table__, CaseORM.__table__, CaseScorableORM.__table__, ExperimentORM.__table__,
              VariantORM.__table__, TrialORM.__table__, TrialMetricORM.__table__]
    Base.metadata.create_all(db, tables=tables)
    sessions = sessionmaker(bind=db, expire_on_commit=False)
    return ImprovementEngine(root, experiment_store=ExperimentStore(sessions), casebook_store=CaseBookStore(sessions))


def report(root: str | Path) -> dict:
    root = Path(root).resolve()
    path = root / "integrated-results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not_run", "root": str(root)}


def run_live_probe(root: str | Path, *, model_config: str | Path = WORKER_ROOT / "worker.models.yaml") -> dict:
    """One readiness call then one baseline/candidate pair on local Ollama only."""
    root = Path(root).resolve()
    model_config = Path(model_config).resolve()
    if not model_config.exists():
        raise FileNotFoundError(f"Worker model configuration not found: {model_config}")
    model_digest = _sha(model_config.read_bytes())
    live_protocol = {"schema_version": "stephanie.sc3-live-protocol.v1",
        "worker_head": _git_head(WORKER_ROOT), "worker_source_sha256": _worker_sources(),
        "worker_model_config_sha256": model_digest, "model_route": "h4_qwen_local",
        "provider": "local Ollama via Worker OpenCode-compatible adapter",
        "policy": "h4.local", "data_policy": "local_only", "expected_outputs": ["ready", "correct"],
        "readiness_calls": 1, "comparison_calls": 2, "max_calls_per_strategy_execution": 2,
        "max_total_model_invocations": 6, "timeout_per_execution_ms": 30000,
        "output_token_limit": 128, "paid_provider": False, "operational_storage": False,
        "write_capable_tools": False, "mission_writes": False}
    live_digest = digest(live_protocol)
    live_protocol["protocol_digest"] = live_digest
    _json(root / "genuine-live-protocol.json", live_protocol)
    runner = Path(__file__).with_name("sc3_worker_live_runner.ts")
    bun = shutil.which("bun")
    if not bun:
        raise RuntimeError("Bun is required to run isolated Worker live probe")
    out_path = root / "genuine-live-results.json"
    child_env = dict(os.environ)
    child_env["WORKER_MODELS_CONFIG"] = str(model_config)
    try:
        completed = subprocess.run([bun, str(runner), str(root / "worker-live"), str(out_path)], cwd=str(WORKER_ROOT),
                                   env=child_env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("isolated local Worker probe exceeded the 120-second total budget; no retry was made") from exc
    if not out_path.exists():
        raise RuntimeError(f"isolated Worker live probe produced no result artifact (exit={completed.returncode}); no retry was made")
    result = json.loads(out_path.read_text(encoding="utf-8"))
    result["protocol_digest"] = live_digest
    result["worker_model_config_sha256_expected"] = model_digest
    invocations = result.get("actual_model_invocations") or 0
    if invocations > 6 or any((attempt.get("usage") or {}).get("invocation_count", 0) > 2 for attempt in result.get("attempts", [])):
        raise ValueError("live model invocation ceiling was exceeded; preserve evidence and stop")
    if completed.returncode != 0:
        result["runner_exit_code"] = completed.returncode
        result["runner_error_category"] = "isolated_runner_error"
    if len(result.get("attempts", [])) >= 3:
        from .integrations import compare_with_praxis
        by_arm = {}
        for label in ("baseline", "candidate"):
            row = next((item for item in result["attempts"] if label.upper() in item["task_id"]), None)
            if row:
                mapped = {"request": row["request"], "response": row["result"], "ledger_record": row.get("ledger_record")}
                score, goal = _score(mapped, PYTHON_ORACLE)
                by_arm[label] = _record(mapped, score, goal, protocol_digest=live_digest,
                                        worker_head=live_protocol["worker_head"], experiment_id="sc3-genuine-live-probe")
        if set(by_arm) == {"baseline", "candidate"}:
            comparison = compare_with_praxis([by_arm["baseline"]], [by_arm["candidate"]],
                praxis_module=type("Praxis", (), {"compare_effectiveness": compare_effectiveness}))
            b_calls = result["attempts"][1].get("usage", {}).get("invocation_count", 0)
            c_calls = result["attempts"][2].get("usage", {}).get("invocation_count", 0)
            confounded = c_calls > b_calls
            result["praxis_live"] = {"comparison": comparison["comparison"], "cohort_size_each": 1,
                "resource_comparison": {"baseline_invocations": b_calls, "candidate_invocations": c_calls,
                    "additional_compute": confounded, "cost_status": "unknown"},
                "recommendation": "WITHHOLD_RESOURCE_CONFOUNDED" if confounded else comparison["comparison"]["decision"],
                "limitation": "one genuine pair is below Praxis's three-runs-per-arm minimum"}
            _json(root / "praxis-live-comparison.json", result["praxis_live"])
    _json(out_path, result)
    summary = {"attempts": len(result.get("attempts", [])), "actual_model_invocations": invocations,
               "stopped_after_readiness_failure": result.get("stopped_after_readiness_failure"),
               "praxis_decision": (result.get("praxis_live") or {}).get("comparison", {}).get("decision"),
               "result_path": str(out_path)}
    _json(root / "genuine-live-summary.json", summary)
    return summary


def recompare_saved(root: str | Path) -> dict:
    """Reapply canonical Praxis to preserved Worker records; performs no execution."""
    root = Path(root).resolve()
    protocol = json.loads((root / "frozen-protocol.json").read_text(encoding="utf-8"))
    if protocol["protocol_digest"] != digest({k: v for k, v in protocol.items() if k != "protocol_digest"}):
        raise ValueError("frozen protocol digest mismatch")
    candidate = validate_candidate(root, protocol)
    worker = json.loads((root / "worker-results.json").read_text(encoding="utf-8"))
    rows = worker["results"]
    comparisons = [_compare(rows, scenario, root, protocol["protocol_digest"], protocol["worker_head"], protocol["experiment_id"])
                   for scenario in ("C1", "C2", "C3", "C4", "C5")]
    c6 = next(row for row in rows if row["request"]["task_id"].startswith("C6_FAIL"))
    c6_record = {"scenario": "C6", "result_status": c6["response"].get("status"),
                 "unresolved": c6["response"].get("unresolved"), "worker_ledger_recorded": bool(c6.get("ledger_record")),
                 "inference_executed": ((c6.get("ledger_record") or {}).get("observed") or {}).get("inference_executed", "UNKNOWN"),
                 "resource_observation": (c6["response"].get("strategy_execution") or {}).get("resources")}
    result = {"schema_version": "stephanie.sc3-experiment-result.v1", "experiment_id": protocol["experiment_id"],
              "protocol_digest": protocol["protocol_digest"], "candidate_id": candidate.candidate_id,
              "worker": {k: worker[k] for k in ("worker_source_commit", "worker_service", "fixture_provider", "storage")},
              "praxis_comparisons": comparisons, "C6_execution_failure": c6_record,
              "C7_invalid_candidate": json.loads((root / "candidate-rejection-c7.json").read_text(encoding="utf-8")),
              "actual_model_invocations": 0, "formal_acceptance": "UNKNOWN", "mission_writes": 0,
              "qualification": "deterministic integration using actual Worker HTTP/service and Praxis callable; scripted provider; no genuine model inference"}
    _json(root / "integrated-results.json", result)
    _retain(root, result, protocol, candidate)
    return {"status": "recompared_saved_records", "execution_count": len(rows),
            "praxis": {item["scenario"]: item["praxis"]["decision"] for item in comparisons},
            "resource_controls": {item["scenario"]: item["resource_comparison"] for item in comparisons}}


def reconcile_interrupted_run(root: str | Path) -> dict:
    """Finish evaluation/retention only from a complete saved Worker run.

    This operation never starts Worker. It refuses partial result sets or a
    changed protocol/source and leaves the run UNKNOWN if evidence is absent.
    """
    root = Path(root).resolve()
    claim_path = root / "sc3-run.claim"
    if not claim_path.is_file():
        raise FileNotFoundError("SC3 run claim is missing")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    protocol = json.loads((root / "frozen-protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol_digest") != digest({k: v for k, v in protocol.items() if k != "protocol_digest"}):
        raise ValueError("frozen protocol digest mismatch")
    if (_worker_sources() != protocol.get("worker_source_sha256")
            or _git_head(WORKER_ROOT) != protocol.get("worker_head")):
        raise ValueError("Worker source changed after the frozen run")
    if claim.get("experiment_id") != protocol.get("experiment_id") or claim.get("protocol_digest") != protocol.get("protocol_digest"):
        raise ValueError("SC3 claim does not match the frozen protocol")
    if claim.get("status") == "completed":
        result_path = root / "integrated-results.json"
        if result_path.is_file() and claim.get("integrated_results_sha256") == _sha(result_path.read_bytes()):
            return {"status": "already_completed", "experiment_id": protocol["experiment_id"]}
        raise ValueError("completed SC3 claim has an invalid result digest")
    if claim.get("status") != "dispatch_may_have_started":
        raise ValueError("SC3 claim is not in a reconcilable state")
    candidate = validate_candidate(root, protocol)
    worker_path = root / "worker-results.json"
    if not worker_path.is_file() or worker_path.stat().st_size > 10_000_000:
        raise RuntimeError("complete Worker result artifact is unavailable; execution remains UNKNOWN")
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    expected_tasks = [row.get("task_id") for row in protocol.get("requests", [])]
    rows = worker.get("results")
    if (worker.get("protocol_digest") != protocol.get("protocol_digest")
            or worker.get("experiment_id") != protocol.get("experiment_id")
            or not isinstance(rows, list) or len(rows) != len(expected_tasks)
            or [row.get("request", {}).get("task_id") for row in rows] != expected_tasks
            or any(not row.get("ledger_record") or row.get("response", {}).get("request_id") != row.get("ledger_record", {}).get("request_id") for row in rows)):
        raise RuntimeError("Worker results do not prove a complete ordered run; execution remains UNKNOWN")
    ledger_path = root / "worker" / "worker-execution-ledger.jsonl"
    storage = worker.get("storage") or {}
    if (not ledger_path.is_file() or storage.get("operational_storage_fallback") is not False
            or Path(storage.get("effective_execution_ledger", "")).resolve() != ledger_path.resolve()):
        raise RuntimeError("isolated Worker ledger identity is not verified; execution remains UNKNOWN")
    ledger_records = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ledger_records.append(json.loads(line))
    # Worker execution summaries wrap the canonical record in ``record``;
    # older/fake fixtures may still store the fields at the top level.
    ledger_by_id: dict[str, dict] = {}
    for item in ledger_records:
        if not isinstance(item, dict):
            raise RuntimeError("Worker ledger contains an invalid record; execution remains UNKNOWN")
        record = item.get("record") if isinstance(item.get("record"), dict) else item
        request_id = record.get("request_id")
        if (not isinstance(request_id, str) or request_id in ledger_by_id
                or ("record" in item and item.get("digest") != digest(record))):
            raise RuntimeError("Worker ledger identity or digest is invalid; execution remains UNKNOWN")
        ledger_by_id[request_id] = record
    row_ids: set[str] = set()
    for row, frozen_request in zip(rows, protocol.get("requests", []), strict=True):
        response = row.get("response") or {}
        saved_record = row.get("ledger_record")
        request_id = response.get("request_id")
        actual_record = ledger_by_id.get(request_id)
        requested = frozen_request.get("execution") or {}
        parameters = frozen_request.get("parameters") or {}
        requested_summary = {
            "engine": requested.get("backend"), "model": requested.get("model"),
            "policy": requested.get("policy"),
            **({"normalization_policy": requested.get("normalization_policy")}
               if requested.get("strategy_id") == "adaptive.v1" else {}),
            "data_policy": parameters.get("data_policy"),
            "reasoning_effort": None, "temperature": None,
            "timeout_ms": parameters.get("timeout_ms"),
            "max_output_tokens": parameters.get("max_output_tokens"),
        }
        normalized_request = {key: value for key, value in frozen_request.items()
                              if key not in {"request_id", "episode_id", "metadata", "prompt"}}
        normalized_request["schema"] = frozen_request.get("schema", "worker.work-request.v1")
        normalized_request["objective"] = str(frozen_request.get("objective") or frozen_request.get("prompt") or "").strip()
        if requested.get("strategy_id"):
            normalized_request["execution"] = {**requested, "backend": "model"}
        expected_input_digest = digest(normalized_request)
        expected_strategy = requested.get("strategy_id")
        expected_normalization = requested.get("normalization_policy")
        if (row.get("request") != frozen_request or request_id in row_ids
                or not isinstance(saved_record, dict) or actual_record is None
                or saved_record != actual_record
                or actual_record.get("task_id") != frozen_request.get("task_id")
                or actual_record.get("operation_id") != protocol.get("experiment_id")
                or actual_record.get("input_digest") != expected_input_digest
                or actual_record.get("requested") != requested_summary
                or ((actual_record.get("resolved") or {}).get("strategy_identity") or {}).get("key") != expected_strategy
                or ((actual_record.get("resolved") or {}).get("strategy_identity") or {}).get("normalization_policy") != expected_normalization
                or actual_record.get("output_digest") != (digest(response["output"]) if "output" in response else None)
                or actual_record.get("input_output_digest_scope") != "public_fixture"):
            raise RuntimeError("saved Worker result is not bound to its exact frozen request and ledger digests; execution remains UNKNOWN")
        row_ids.add(request_id)
    comparisons = [_compare(rows, scenario, root, protocol["protocol_digest"],
                            protocol["worker_head"], protocol["experiment_id"])
                   for scenario in ("C1", "C2", "C3", "C4", "C5")]
    c6 = next((row for row in rows if row["request"]["task_id"].startswith("C6_FAIL")), None)
    c7_path = root / "candidate-rejection-c7.json"
    if c6 is None or not c7_path.is_file():
        raise RuntimeError("frozen failure/invalid-candidate controls are missing; execution remains UNKNOWN")
    c6_record = {"scenario": "C6", "result_status": c6["response"].get("status"),
        "unresolved": c6["response"].get("unresolved"), "worker_ledger_recorded": bool(c6.get("ledger_record")),
        "inference_executed": ((c6.get("ledger_record") or {}).get("observed") or {}).get("inference_executed", "UNKNOWN"),
        "resource_observation": (c6["response"].get("strategy_execution") or {}).get("resources")}
    result = {"schema_version": "stephanie.sc3-experiment-result.v1",
        "experiment_id": protocol["experiment_id"], "protocol_digest": protocol["protocol_digest"],
        "candidate_id": candidate.candidate_id,
        "worker_ledger_sha256": _sha(ledger_path.read_bytes()),
        "worker_ledger_record_count": len(ledger_records),
        "worker": {k: worker[k] for k in ("worker_source_commit", "worker_service", "fixture_provider", "storage")},
        "praxis_comparisons": comparisons, "C6_execution_failure": c6_record,
        "C7_invalid_candidate": json.loads(c7_path.read_text(encoding="utf-8")),
        "actual_model_invocations": 0, "formal_acceptance": "UNKNOWN", "mission_writes": 0,
        "qualification": "reconciled complete saved Worker results; deterministic scripted provider; no Worker retry and no genuine model inference"}
    _json(root / "integrated-results.json", result)
    _retain(root, result, protocol, candidate)
    claim.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                  "integrated_results_sha256": _sha((root / "integrated-results.json").read_bytes()),
                  "reconciled_without_worker_retry": True})
    _json(claim_path, claim)
    return {"status": "reconciled", "experiment_id": protocol["experiment_id"],
            "worker_requests": len(rows), "praxis_scenarios": {x["scenario"]: x["praxis"]["decision"] for x in comparisons}}


def retrieve_experience(root: str | Path, query: str) -> list[dict]:
    root = Path(root).resolve()
    engine = _local_memory_engine(root)
    book = engine.casebook_store
    book_id = book.get_id_by_name("stephanie-sc3-integrated-experiments")
    if book_id is None:
        return []
    tokens = {word.lower() for word in query.split() if len(word) > 2}
    found = []
    for summary in book.list_case_summaries(book_id):
        meta = summary.get("meta") or {}
        item = meta.get("integrated_experiment") or {}
        searchable = " ".join(str(value) for value in [summary.get("name"), summary.get("description"),
                            item.get("outcome"), item.get("limitations"), item.get("original_observation"),
                            item.get("recommendation"), item.get("scenario")]).lower()
        overlap = len(tokens & set(searchable.split()))
        if overlap:
            found.append({"name": summary.get("name"), "outcome": item.get("outcome"),
                          "scenario": item.get("scenario"), "reuse_eligible": meta.get("reuse_eligible"),
                          "promotion_status": meta.get("promotion_status"), "matched_terms": overlap})
    return sorted(found, key=lambda row: (-row["matched_terms"], str(row["name"])))
