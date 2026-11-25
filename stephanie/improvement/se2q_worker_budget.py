"""Single evidence-driven Worker budget follow-up; hard cap of four calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .engine import digest
from .evolution02q import DATA, SCHEMA_FILE, TASKS, _read, _sha, _task_prompt
from .worker_harness import WorkerHarnessClient, validate_worker_hypothesis
from .evolution02q import ReadinessAnswer

MANIFEST = DATA / "worker-budget-followup-manifest.json"
RESULTS = DATA / "worker-1024-results.json"
PORT = 17319


def freeze() -> dict:
    """Pre-register a two-task, one-factor output-budget diagnostic."""
    prior_manifest = _read(DATA / "frozen-manifest.json")
    prior_results = DATA / "readiness-results.json"
    if not prior_results.exists():
        raise RuntimeError("the initial diagnostic must be complete before its budget follow-up")
    source = Path(__file__)
    adapter = Path(__file__).with_name("worker_harness.py")
    prompt_source = Path(__file__).with_name("evolution02q.py")
    parent_manifest_path = DATA / "frozen-manifest.json"
    payload = {
        "protocol":"STEPHANIE-EVOLUTION-02Q/worker-budget/v1",
        "parent_protocol_digest":prior_manifest["protocol_digest"],
        "parent_manifest_sha256":_sha(parent_manifest_path),
        "prior_results_sha256":_sha(prior_results),
        "tasks_sha256":_sha(TASKS), "schema_sha256":_sha(SCHEMA_FILE),
        "task_ids":["se2q-dev-01-observation-boundary", "se2q-dev-02-representation-boundary"],
        "arms":{"prior":{"strategy":"adaptive.v1", "output_tokens":512},
                "followup":{"strategy":"adaptive.v1", "output_tokens":1024}},
        "held_constant":["task prompt", "schema", "local model", "provider", "policy", "strategy", "temperature"],
        "model":{"provider":"ollama", "name":"ornith-1.5:9b", "local_id":"e5df7dcdd8a2"},
        "worker":{"route":"se2q_ornith_local", "policy":"se2q.local", "strategy":"adaptive.v1", "normalization":"json-fence.v1", "timeout_ms":60000, "port":PORT},
        "max_calls_per_task":2, "planned_max_invocations":4,
        "remaining_call_authorization_after_run1":4,
        "source_sha256":{str(path):_sha(path) for path in (source,adapter,prompt_source,TASKS, SCHEMA_FILE)},
        "classification":"development-only follow-up; a single output-budget adjustment based on an observed 512-token truncated Worker response",
    }
    payload["protocol_digest"] = digest(payload)
    if MANIFEST.exists() and _read(MANIFEST) != payload:
        raise RuntimeError("budget follow-up already frozen and differs")
    MANIFEST.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def run() -> dict:
    manifest = _read(MANIFEST)
    if manifest["protocol_digest"] != digest({k:v for k,v in manifest.items() if k!="protocol_digest"}):
        raise RuntimeError("follow-up manifest digest mismatch")
    if RESULTS.exists():
        raise RuntimeError("budget follow-up already executed; no retries")
    parent_manifest_path = DATA / "frozen-manifest.json"
    if _sha(parent_manifest_path) != manifest["parent_manifest_sha256"] or _read(parent_manifest_path)["protocol_digest"] != manifest["parent_protocol_digest"]:
        raise RuntimeError("parent readiness manifest changed")
    for name, expected in manifest["source_sha256"].items():
        if _sha(Path(name)) != expected:
            raise RuntimeError(f"follow-up source changed: {name}")
    if _sha(DATA/"readiness-results.json") != manifest["prior_results_sha256"]:
        raise RuntimeError("prior readiness result changed")
    client=WorkerHarnessClient(f"http://127.0.0.1:{PORT}",model="se2q_ornith_local",policy="se2q.local",output_tokens=1024)
    health=client.health()
    if not health.get("version") or not any(row.get("id")=="model" and row.get("available") for row in health.get("executors",[])):
        raise RuntimeError("isolated follow-up Worker is not healthy")
    tasks={t["task_id"]:t for t in _read(TASKS)}
    schema=_read(SCHEMA_FILE)
    records=[]
    for task_id in manifest["task_ids"]:
        task=tasks[task_id]
        try:
            result=client.run(task_id=task_id,objective=_task_prompt(task),output_schema=schema)
            allowed_ids={item["evidence_id"] for item in task["observations"]}
            validation=validate_worker_hypothesis(result,ReadinessAnswer,allowed_evidence_ids=allowed_ids)
            row={"task_id":task_id,"status":"READY" if validation["eligible"] else "FAILED","work_result":result,"stephanie_validation":validation}
        except Exception as exc:
            row={"task_id":task_id,"status":"FAILED","error":f"{type(exc).__name__}: {exc}"}
        records.append(row)
        RESULTS.write_text(json.dumps(records,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
    invocations=sum(len(r.get("work_result",{}).get("strategy_execution",{}).get("invocations",[])) for r in records)
    report={"protocol_digest":manifest["protocol_digest"],"n_tasks":len(records),"ready":sum(r["status"]=="READY" for r in records),"failed":sum(r["status"]!="READY" for r in records),"strategy_invocations":invocations,"within_four_call_plan":invocations<=4,"results":str(RESULTS)}
    (DATA/"worker-budget-followup-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("freeze","run")); ns=parser.parse_args()
    print(json.dumps(freeze() if ns.command=="freeze" else run(),indent=2,sort_keys=True))
