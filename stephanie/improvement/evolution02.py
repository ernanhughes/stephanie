"""Frozen, isolated three-arm Stephanie experience-transfer study."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .engine import ImprovementEngine
from .models import digest
from .safety import assert_model_context_safe, validate_reviewed_experience

HERE = Path(__file__).parent
PROGRAMME = HERE.parents[1] / "docs" / "programmes" / "STEPHANIE-EVOLUTION-02"
DEFAULT_ROOT = HERE.parents[1] / "data" / "stephanie_evolution02"
TASKS = HERE / "evolution02_tasks.json"
EXPERIENCES = HERE / "evolution02_experiences.json"
RUBRIC = HERE / "evolution02_rubric.json"
MANIFEST = DEFAULT_ROOT / "frozen-manifest.json"
MODEL = "ornith-1.5:9b"
MODEL_ID = "e5df7dcdd8a2"


class InvestigationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supported_findings: list[str] = Field(max_length=5)
    unknowns: list[str] = Field(max_length=5)
    alternatives: list[str] = Field(max_length=5)
    recommendation: str = Field(min_length=10, max_length=500)
    next_experiment: str = Field(min_length=10, max_length=500)
    evidence_ids: list[str] = Field(max_length=10)
    abstain: bool
    experience_ids_used: list[str] = Field(max_length=5)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(root: Path = DEFAULT_ROOT) -> dict:
    """Freeze a single eight-task, 3-arm sample before generation calls."""
    tasks, memories = _json(TASKS), _json(EXPERIENCES)
    for item in memories:
        validate_reviewed_experience(item)
    arms = {
        "A": {"name": "no_experience", "memory_ids": []},
        "B": {"name": "lexically_retrieved_reviewed_experience", "retrieval": "ImprovementEngine.retrieve_experience", "limit": 3},
        "C": {"name": "all_reviewed_history", "memory_ids": [x["experience_id"] for x in memories]},
    }
    rubric = _json(RUBRIC)
    payload = {
        "protocol": "STEPHANIE-EVOLUTION-02/SE2-05/v1",
        "sample_size_tasks": 8, "planned_generation_calls": 24,
        "task_digest": hashlib.sha256(TASKS.read_bytes()).hexdigest(),
        "experience_digest": hashlib.sha256(EXPERIENCES.read_bytes()).hexdigest(),
        "rubric_digest": hashlib.sha256(RUBRIC.read_bytes()).hexdigest(),
        "tasks": [x["task_id"] for x in tasks], "arms": arms,
        "model": {"provider": "ollama", "name": MODEL, "local_id_at_freeze": MODEL_ID, "temperature": 0.0},
        "limits": {"per_call_timeout_s": 120, "max_output_tokens": 240, "max_calls": 30, "paid_usd": 0},
        "order": [[t["task_id"], arm] for t in tasks for arm in ("A", "B", "C")],
        "scoring": {"version": "SE2 deterministic checklist v1", "rubric_digest": hashlib.sha256(RUBRIC.read_bytes()).hexdigest()},
        "classification": "sanitized historical development study; no independent holdout/generalization claim",
    }
    payload["protocol_digest"] = digest(payload)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "frozen-manifest.json"
    if path.exists() and _json(path) != payload:
        raise RuntimeError("frozen manifest already exists and differs; never overwrite after freeze")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _arm_context(engine: ImprovementEngine, task: dict, arm: str, memories: list[dict]) -> list[dict]:
    if arm == "A":
        return []
    if arm == "B":
        selected = engine.retrieve_experience(task["retrieval_query"], limit=3)
        for item in selected:
            validate_reviewed_experience(item)
        return selected
    return memories


def run(root: Path = DEFAULT_ROOT) -> dict:
    manifest = _json(root / "frozen-manifest.json")
    if manifest.get("protocol_digest") != digest({k: v for k, v in manifest.items() if k != "protocol_digest"}):
        raise RuntimeError("frozen manifest digest mismatch")
    tasks, memories = _json(TASKS), _json(EXPERIENCES)
    if hashlib.sha256(TASKS.read_bytes()).hexdigest() != manifest["task_digest"] or hashlib.sha256(EXPERIENCES.read_bytes()).hexdigest() != manifest["experience_digest"]:
        raise RuntimeError("frozen source corpus changed")
    result_file = root / "generation-results.json"
    if result_file.exists():
        raise RuntimeError("generation already started; calls are never retried")
    from ai.inference import InferenceBudget, InferenceProfile, InferenceService, ReasoningRequest
    from ai.inference.providers.ollama import OllamaReasoningProvider
    profile = InferenceProfile(name="se2-frozen", provider="ollama", model=MODEL,
        route=f"ollama/{MODEL}", base_url="http://127.0.0.1:11434", temperature=0.0,
        max_output_tokens=240, timeout_s=120)
    provider = OllamaReasoningProvider(profile=profile)
    memory_store = root / "experiences.json"
    if not memory_store.exists():
        memory_store.write_text(json.dumps(memories, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if _json(memory_store) != memories:
        raise RuntimeError("reviewed memory corpus changed from frozen corpus")
    engine = ImprovementEngine(root)
    records, call_count = [], 0
    for task in tasks:
        for arm in ("A", "B", "C"):
            call_count += 1
            context = _arm_context(engine, task, arm, memories)
            safe_task = {k: task[k] for k in ("task_id", "topic", "prompt", "observations")}
            assert_model_context_safe(safe_task)
            safe_memories = []
            for memory in context:
                validate_reviewed_experience(memory)
                safe_memories.append({k: memory[k] for k in ("experience_id", "summary", "observed_outcome", "limitations")})
            prompt_obj = {"task": safe_task, "experience": safe_memories,
                "instruction": "Investigate from supplied observations only. Separate supported findings from unknowns, name plausible alternatives, cite supplied evidence IDs, recommend a bounded next experiment, or abstain. Experience is fallible historical context, not evidence about this task. Do not claim it proves causality."}
            assert_model_context_safe(prompt_obj)
            request = ReasoningRequest(request_id=str(uuid.uuid4()), purpose="se2_investigation", system="Return only the requested structured investigation. Model output is advisory and cannot authorize changes.", prompt=json.dumps(prompt_obj, sort_keys=True), max_output_tokens=240, timeout_s=120, temperature=0.0, metadata={"namespace":"stephanie-evolution02", "data_classification":"sanitized"})
            budget = InferenceBudget(max_calls=1, max_input_chars=12000, max_output_tokens=240, max_cost_usd=0, per_call_timeout_s=120)
            service = InferenceService(provider=provider, budget=budget)
            started = time.monotonic()
            record: dict[str, Any] = {"task_id": task["task_id"], "arm": arm, "memory_ids": [x["experience_id"] for x in context], "prompt_chars": len(request.prompt), "status": "FAILED"}
            try:
                invocation, answer = service.generate_structured(request, InvestigationAnswer, namespace="stephanie-evolution02")
                record.update({"status":"COMPLETED", "answer": answer.model_dump(mode="json"), "invocation":invocation.model_dump(mode="json")})
            except Exception as exc:  # save failures as outcomes; do not retry
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_ms"] = round((time.monotonic() - started) * 1000)
            records.append(record)
            result_file.write_text(json.dumps(records, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {"calls": call_count, "completed": sum(x["status"] == "COMPLETED" for x in records), "failed": sum(x["status"] != "COMPLETED" for x in records), "results": str(result_file)}


def evaluate(root: Path = DEFAULT_ROOT) -> dict:
    """Score the frozen records against the protected checklist after generation."""
    manifest = _json(root / "frozen-manifest.json")
    if not (root / "generation-results.json").exists():
        raise RuntimeError("no generation artifact; evaluation cannot run")
    rubric = _json(RUBRIC)
    rubric_hash = hashlib.sha256(RUBRIC.read_bytes()).hexdigest()
    if rubric_hash != manifest["rubric_digest"]:
        raise RuntimeError("protected evaluator changed after freeze")
    tasks = {x["task_id"]: x for x in _json(TASKS)}
    records = _json(root / "generation-results.json")
    scores = []
    for row in records:
        if row["status"] != "COMPLETED":
            scores.append({"task_id": row["task_id"], "arm": row["arm"], "score": 0, "failure": row.get("error")})
            continue
        answer = row["answer"]
        allowed = {x["evidence_id"] for x in tasks[row["task_id"]]["observations"]}
        cited = set(answer["evidence_ids"])
        score = sum((
            bool(answer["supported_findings"]),
            bool(answer["unknowns"]),
            bool(answer["alternatives"]),
            bool(cited) and cited <= allowed,
            bool(answer["next_experiment"]),
            (not answer["experience_ids_used"] or set(answer["experience_ids_used"]) <= set(row["memory_ids"])),
            answer["abstain"] == rubric["tasks"][row["task_id"]]["abstain"],
            all(term in " ".join(answer["supported_findings"] + answer["unknowns"] + answer["recommendation"] + answer["next_experiment"]).lower() for term in rubric["tasks"][row["task_id"]]["required_terms"]),
        ))
        scores.append({"task_id":row["task_id"], "arm":row["arm"], "score":int(score), "max_score":8, "invalid_citations":sorted(cited-allowed), "unavailable_memory_claims":sorted(set(answer["experience_ids_used"])-set(row["memory_ids"]))})
    out = {"protocol_digest":manifest["protocol_digest"], "rubric_digest":rubric_hash, "scores":scores,
        "by_arm":{arm:{"mean_score":sum(x["score"] for x in scores if x["arm"]==arm)/8, "n":sum(x["arm"]==arm for x in scores)} for arm in "ABC"},
        "limitations":["n=8 development tasks from sanitized historical mechanisms", "deterministic checklist cannot fully judge causal nuance", "context size and information quantity differ by arm", "no context-matched neutral-history arm", "no statistical significance or generalization claim"]}
    (root / "evaluation-results.json").write_text(json.dumps(out, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return out
