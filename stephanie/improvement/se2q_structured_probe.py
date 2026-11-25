"""One-call direct qualification of AI InferenceService.generate_structured."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from .engine import digest
from .evolution02q import DATA, MODEL, MODEL_ID, ROOT, TASKS, ReadinessAnswer, _read, _sha, _task_prompt
from .safety import assert_model_context_safe

MANIFEST=DATA/"structured-api-probe-manifest.json"
RESULT=DATA/"structured-api-probe.json"
SCHEMA=ROOT/"stephanie/improvement/evolution02q_schema_v2.json"
TASK_ID="se2q-dev-03-structural-vs-semantic"


def freeze() -> dict:
    tasks={t["task_id"]:t for t in _read(TASKS)}
    task=tasks[TASK_ID]
    prompt=_task_prompt(task)
    ai=Path("C:/Projects/ai/src/ai/inference")
    source_paths=[Path(__file__), ROOT/"stephanie/improvement/evolution02q.py", ROOT/"stephanie/improvement/worker_harness.py",
        TASKS,SCHEMA,ai/"service.py",ai/"dto.py",ai/"providers/ollama.py"]
    parent=_read(DATA/"frozen-manifest.json")
    previous=_read(DATA/"readiness-results.json")
    payload={
        "protocol":"STEPHANIE-EVOLUTION-02Q/direct-structured-parser/v1",
        "parent_protocol_digest":parent["protocol_digest"],
        "previous_results_sha256":_sha(DATA/"readiness-results.json"),
        "task_id":TASK_ID,"task_prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
        "tasks_sha256":_sha(TASKS),"schema_sha256":_sha(SCHEMA),
        "model":{"provider":"ollama","name":MODEL,"local_id":MODEL_ID,"temperature":0},
        "route":{"api":"shared ai.inference.InferenceService.generate_structured","provider_api":"ollama /api/generate","native_schema":"format JSON Schema"},
        "limits":{"planned_model_calls":1,"max_calls":1,"max_output_tokens":512,"timeout_s":60,"paid_usd":0},
        "sources":{str(path):_sha(path) for path in source_paths},
        "classification":"development readiness parser-path probe; previously exposed development task; no effectiveness or holdout claim",
    }
    payload["protocol_digest"]=digest(payload)
    if MANIFEST.exists() and _read(MANIFEST)!=payload:
        raise RuntimeError("structured-parser protocol already frozen and differs")
    MANIFEST.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def run() -> dict:
    manifest=_read(MANIFEST)
    if manifest["protocol_digest"]!=digest({k:v for k,v in manifest.items() if k!="protocol_digest"}):
        raise RuntimeError("structured-parser protocol digest mismatch")
    if manifest["parent_protocol_digest"]!=_read(DATA/"frozen-manifest.json")["protocol_digest"]:
        raise RuntimeError("parent readiness protocol changed")
    if manifest["previous_results_sha256"]!=_sha(DATA/"readiness-results.json"):
        raise RuntimeError("parent readiness results changed")
    for name,expected in manifest["sources"].items():
        if _sha(Path(name))!=expected: raise RuntimeError(f"frozen parser probe source changed: {name}")
    if RESULT.exists(): raise RuntimeError("structured parser probe already executed; no retry")
    task=next(t for t in _read(TASKS) if t["task_id"]==TASK_ID)
    prompt=_task_prompt(task)
    assert_model_context_safe({"prompt":prompt,"observations":task["observations"]})
    from ai.inference import InferenceBudget,InferenceProfile,InferenceService,ReasoningRequest
    from ai.inference.providers.ollama import OllamaReasoningProvider
    profile=InferenceProfile(name="se2q-generate-structured",provider="ollama",model=MODEL,
        route=f"ollama/{MODEL}",base_url="http://127.0.0.1:11434",temperature=0,max_output_tokens=512,timeout_s=60)
    traces=[]
    service=InferenceService(provider=OllamaReasoningProvider(profile=profile),
        budget=InferenceBudget(max_calls=1,max_input_chars=12000,max_output_tokens=512,max_cost_usd=0,per_call_timeout_s=60),
        trace_sink=traces.append)
    req=ReasoningRequest(request_id=str(uuid.uuid4()),purpose="se2q_shared_structured_path_probe",
        system="Investigate only the supplied development evidence. Output is advisory, not authorization.",
        prompt=prompt,response_schema=_read(SCHEMA),max_output_tokens=512,timeout_s=60,temperature=0,
        metadata={"namespace":"stephanie-se2q-structured","data_classification":"synthetic_development"})
    start=time.monotonic()
    row={"task_id":TASK_ID,"route":"InferenceService.generate_structured","status":"FAILED"}
    try:
        result,answer=service.generate_structured(req,ReadinessAnswer,namespace="stephanie-se2q-structured")
        allowed={x["evidence_id"] for x in task["observations"]}
        if not answer.evidence_ids or not set(answer.evidence_ids).issubset(allowed):
            raise ValueError("empty or unavailable evidence citation")
        row.update({"status":"READY","raw_response":result.content,"response_fingerprint":result.response_fingerprint,
            "provider":result.provider,"model":result.model,"usage":result.usage.model_dump(mode="json"),
            "raw_metadata":result.raw_metadata,"answer":answer.model_dump(mode="json"),
            "evidence_ids_valid":True})
    except Exception as exc:
        row["error"]=f"{type(exc).__name__}: {exc}"
    row["latency_ms"]=round((time.monotonic()-start)*1000)
    row["trace"]=traces[-1] if traces else None
    row["protocol_digest"]=manifest["protocol_digest"]
    RESULT.write_text(json.dumps(row,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
    return row


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("freeze","run"));args=parser.parse_args()
    print(json.dumps(freeze() if args.command=="freeze" else run(),indent=2,sort_keys=True,default=str))
