"""SE2Q frozen, bounded direct-AI vs Worker execution readiness check."""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from pathlib import Path

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .engine import digest
from .safety import assert_model_context_safe
from .worker_harness import WorkerHarnessClient, validate_worker_hypothesis

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "stephanie_evolution02q"
TASKS = ROOT / "stephanie" / "improvement" / "evolution02q_tasks.json"
SCHEMA_FILE = ROOT / "stephanie" / "improvement" / "evolution02q_schema.json"
MANIFEST = DATA / "frozen-manifest.json"
RESULTS = DATA / "readiness-results.json"
WORKER_URL = "http://127.0.0.1:17318"
MODEL = "ornith-1.5:9b"
MODEL_ID = "e5df7dcdd8a2"
WORKER_MODELS_CONFIG = DATA / "worker" / "worker.models.yaml"
WORKER_CONFIG = DATA / "worker" / "worker.json"


class ReadinessAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(min_length=1, max_length=2)
    unknowns: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(min_length=1, max_length=2)
    next_step: str = Field(min_length=10, max_length=300)
    evidence_ids: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(min_length=1, max_length=4)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_isolated_worker_config() -> tuple[Path, Path]:
    """Derive a run-local model route/config without modifying Worker files."""
    source = Path("C:/Projects/worker/worker.models.yaml")
    yaml_text = source.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^policies:\r?\n", yaml_text))
    if len(matches) != 1:
        raise RuntimeError("Worker models config shape changed; inspect before deriving isolated route")
    route = (
        "  se2q_ornith_local:\n"
        "    provider: h4_local\n"
        f"    model: {MODEL}\n"
        "    economic_tier: free\n"
        "    data_policy: local_only\n"
        "    verification: catalogue_listed\n"
    )
    marker = matches[0]
    isolated = yaml_text[:marker.start()] + route + yaml_text[marker.start():] + (
        "\n  se2q.local:\n"
        "    strategy: ordered_fallback\n"
        "    routes: [se2q_ornith_local]\n"
        "    allow_direct_paid: false\n"
        "    attempts_per_route: 1\n"
        "    required_data_policy: local_only\n"
    )
    WORKER_MODELS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    isolated_config = {
        "server":{"host":"127.0.0.1","port":17318},
        "executor":{"default":"model"},
        "executors":{"model":{"type":"model"}},
        "model_gateways":{}, "profiles":{},
    }
    for path, content in ((WORKER_MODELS_CONFIG, isolated), (WORKER_CONFIG, json.dumps(isolated_config, indent=2)+"\n")):
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"isolated Worker config already exists and differs: {path}")
        path.write_text(content, encoding="utf-8")
    return WORKER_MODELS_CONFIG, WORKER_CONFIG


def freeze() -> dict:
    """Freeze dev prompts, schema, order, model, route and source identities."""
    isolated_models, isolated_worker = prepare_isolated_worker_config()
    worker = Path("C:/Projects/worker")
    ai = Path("C:/Projects/ai/src/ai/inference")
    source_paths = [
        ROOT / "stephanie/improvement/evolution02q.py",
        ROOT / "stephanie/improvement/worker_harness.py",
        ai / "service.py", ai / "providers/ollama.py", ai / "dto.py",
        worker / "src/strategies/runtime.ts", worker / "src/strategies/catalog.ts",
        worker / "src/executors/model.ts", worker / "src/models/providers/opencode.ts",
        worker / "src/models/runtime/config.ts", worker / "src/server/server.ts",
        worker / "worker.models.yaml",
        isolated_models, isolated_worker,
    ]
    payload = {
        "protocol": "STEPHANIE-EVOLUTION-02Q/readiness/v1",
        "tasks_sha256": _sha(TASKS), "schema_sha256": _sha(SCHEMA_FILE),
        "task_ids": [x["task_id"] for x in _read(TASKS)],
        "order": [[task["task_id"], arm] for arm in ("A_DIRECT_AI", "B_WORKER_ADAPTIVE") for task in _read(TASKS)],
        "arms": {
            "A_DIRECT_AI": {"model": MODEL, "provider": "ollama", "max_output_tokens": 512, "timeout_s": 60, "temperature": 0, "calls_per_task": 1, "schema_mode": "ollama_format_json_schema", "response_preserved": True},
            "B_WORKER_ADAPTIVE": {"model": MODEL, "worker_route": "se2q_ornith_local", "policy": "se2q.local", "strategy": "adaptive.v1", "normalization": "json-fence.v1", "max_output_tokens_per_strategy": 512, "timeout_s": 60, "max_calls_per_task": 2, "schema_mode": "worker_structural_json_schema"},
        },
        "sample_size": 3,
        "planned_max_model_invocations": 9,
        "absolute_authorized_ceiling": 10,
        "limits": {"timeout_per_call_s": 120, "paid_usd": 0, "tools": 0, "worker_port": 17318, "storage": str(DATA.resolve())},
        "model_local_tag_id_at_freeze": MODEL_ID,
        "sources": {str(path): _sha(path) for path in source_paths},
        "classification": "development readiness only; tasks are distinct from frozen SE2 task IDs; no effectiveness inference",
    }
    payload["protocol_digest"] = digest(payload)
    DATA.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and _read(MANIFEST) != payload:
        raise RuntimeError("SE2Q manifest already frozen and differs; do not overwrite")
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _task_prompt(task: dict) -> str:
    return ("Return a concise JSON hypothesis investigation. Separate observed facts from unknowns. "
            "Cite only the supplied evidence IDs. Do not infer an unsupported cause. "
            "Use no more than two findings and two unknowns.\n\n" + task["prompt"] +
            "\n\nVisible evidence:\n" + json.dumps(task["observations"], sort_keys=True))


def run(worker_url: str = WORKER_URL) -> dict:
    manifest = _read(MANIFEST)
    if manifest.get("protocol_digest") != digest({k: v for k, v in manifest.items() if k != "protocol_digest"}):
        raise RuntimeError("SE2Q manifest digest mismatch")
    if _sha(TASKS) != manifest["tasks_sha256"] or _sha(SCHEMA_FILE) != manifest["schema_sha256"]:
        raise RuntimeError("frozen SE2Q tasks/schema changed")
    for name, path in manifest["sources"].items():
        if _sha(Path(name)) != path:
            raise RuntimeError(f"frozen source changed: {name}")
    if RESULTS.exists():
        raise RuntimeError("SE2Q readiness run already exists; never retry to improve results")
    schema = _read(SCHEMA_FILE)
    tasks = _read(TASKS)
    # Confirm the isolated Worker is alive before any model call.
    client = WorkerHarnessClient(worker_url, model="se2q_ornith_local", policy="se2q.local")
    health = client.health()
    if not health.get("version") or not any(row.get("id") == "model" and row.get("available") is True for row in health.get("executors", [])):
        raise RuntimeError("isolated Worker health did not qualify the model executor")
    records = []
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    # Direct AI lane: call the shared provider with its native JSON Schema
    # setting, but retain ReasoningResult even when Stephanie parsing fails.
    from ai.inference import InferenceBudget, InferenceProfile, InferenceService, ReasoningRequest
    from ai.inference.providers.ollama import OllamaReasoningProvider
    profile = InferenceProfile(name="se2q-direct", provider="ollama", model=MODEL,
        route=f"ollama/{MODEL}", base_url="http://127.0.0.1:11434", temperature=0,
        max_output_tokens=512, timeout_s=60)
    for task in tasks:
        prompt = _task_prompt(task)
        assert_model_context_safe({"prompt": prompt, "observations": task["observations"]})
        traces = []
        service = InferenceService(provider=OllamaReasoningProvider(profile=profile),
            budget=InferenceBudget(max_calls=1, max_input_chars=12000, max_output_tokens=512, max_cost_usd=0, per_call_timeout_s=60),
            trace_sink=traces.append)
        req = ReasoningRequest(request_id=str(uuid.uuid4()), purpose="se2q_development_readiness",
            system="Investigate only the visible development evidence. Model output is advisory, never authorization.",
            prompt=prompt, response_schema=schema, max_output_tokens=512, timeout_s=60,
            temperature=0, metadata={"namespace":"stephanie-se2q", "data_classification":"synthetic_development"})
        start = time.monotonic()
        record = {"task_id": task["task_id"], "arm":"A_DIRECT_AI", "prompt_chars":len(prompt), "status":"FAILED"}
        try:
            result = service.generate(req, namespace="stephanie-se2q")
            record.update({"raw_response":result.content, "provider":"ollama", "model":result.model,
                "usage":result.usage.model_dump(mode="json"), "raw_metadata":result.raw_metadata,
                "response_fingerprint":result.response_fingerprint, "trace":traces[-1] if traces else None})
            parsed = ReadinessAnswer.model_validate_json(result.content)
            valid_ids = {item["evidence_id"] for item in task["observations"]}
            if not parsed.evidence_ids or not set(parsed.evidence_ids).issubset(valid_ids):
                raise ValueError("missing or unknown evidence IDs")
            record.update({"status":"READY", "answer":parsed.model_dump(mode="json")})
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["trace"] = traces[-1] if traces else None
        record["latency_ms"] = round((time.monotonic()-start)*1000)
        records.append(record); RESULTS.write_text(json.dumps(records, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")

    # Worker lane: public Worker Work API, fixed local-only route, adaptive.v1.
    for task in tasks:
        start = time.monotonic()
        record = {"task_id":task["task_id"], "arm":"B_WORKER_ADAPTIVE", "status":"FAILED"}
        try:
            work = client.run(task_id=task["task_id"], objective=_task_prompt(task), output_schema=schema)
            allowed_ids = {item["evidence_id"] for item in task["observations"]}
            eligibility = validate_worker_hypothesis(work, ReadinessAnswer, allowed_evidence_ids=allowed_ids)
            record.update({"status":"READY" if eligibility["eligible"] else "FAILED", "worker_result":work,
                           "stephanie_validation":eligibility})
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_ms"] = round((time.monotonic()-start)*1000)
        records.append(record); RESULTS.write_text(json.dumps(records, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")
    return report()


def report() -> dict:
    rows = _read(RESULTS)
    by_arm = {}
    invocations = 0
    for arm in ("A_DIRECT_AI", "B_WORKER_ADAPTIVE"):
        subset = [x for x in rows if x["arm"] == arm]
        ready = sum(x["status"] == "READY" for x in subset)
        if arm == "B_WORKER_ADAPTIVE":
            invocations += sum(len(x.get("worker_result", {}).get("strategy_execution", {}).get("invocations", [])) for x in subset)
        else:
            invocations += len(subset)
        by_arm[arm] = {"n":len(subset), "ready":ready, "failed":len(subset)-ready,
                       "latency_ms_total":sum(x.get("latency_ms",0) for x in subset)}
    result = {"protocol_digest":_read(MANIFEST)["protocol_digest"], "by_arm":by_arm,
              "model_invocations":invocations, "within_ten_call_ceiling":invocations <= 10,
              "readiness_only":True, "raw_response_persistence":"direct AI response bodies and full Worker WorkResults retained"}
    (DATA/"readiness-report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result
