"""Thin, fail-closed client for an already isolated Worker /v1/work runtime.

Stephanie owns hypotheses and eligibility. Worker owns execution strategy,
bounded normalization/correction, and execution evidence.
"""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from typing import Any

from .safety import assert_model_context_safe


class WorkerHarnessError(RuntimeError):
    pass


class WorkerHarnessClient:
    def __init__(self, base_url: str, *, model: str, policy: str,
                 timeout_s: int = 60, output_tokens: int = 512):
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Worker adapter requires an explicit loopback HTTP endpoint")
        if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Worker base URL must contain only a loopback authority")
        if not model or not policy or not 1 <= timeout_s <= 60 or not 1 <= output_tokens <= 2048:
            raise ValueError("Worker route and bounded resource settings are required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.policy = policy
        self.timeout_s = timeout_s
        self.output_tokens = output_tokens

    def health(self) -> dict:
        return self._request("GET", "/health")

    def run(self, *, task_id: str, objective: str, output_schema: dict,
            data_policy: str = "local_only") -> dict:
        prompt_payload = {"task_id": task_id, "objective": objective,
                          "output_schema": output_schema,
                          "data_policy": data_policy}
        assert_model_context_safe(prompt_payload)
        if data_policy != "local_only":
            raise ValueError("SE2Q diagnostic requests are restricted to local_only data policy")
        request_body = {
            "schema": "worker.work-request.v1",
            "app": "stephanie-se2q",
            "task_id": task_id,
            "objective": objective,
            "execution": {
                "backend": "model", "model": self.model, "policy": self.policy,
                "strategy_id": "adaptive.v1", "normalization_policy": "json-fence.v1",
            },
            "parameters": {
                "data_policy": "local_only", "timeout_ms": self.timeout_s * 1000,
                "max_output_tokens": self.output_tokens, "temperature": 0,
                "output_schema": output_schema,
            },
            "options": {"acceptance": "human"},
            "metadata": {"stage_id": "SE2Q-READINESS", "data_classification": "synthetic_development"},
        }
        return self._request("POST", "/v1/work", request_body)

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        request = Request(self.base_url + path, data=data, method=method,
                          headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with urlopen(request, timeout=self.timeout_s + 5) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise WorkerHarnessError(f"Worker HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise WorkerHarnessError(f"Worker transport/protocol error: {type(exc).__name__}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise WorkerHarnessError("Worker response root must be an object")
        return decoded


def validate_worker_hypothesis(result: dict, schema_validator, *, allowed_evidence_ids: set[str]) -> dict[str, Any]:
    """Apply Stephanie's independent shape/evidence gate to WorkResult output."""
    strategy = result.get("strategy_execution")
    if not isinstance(strategy, dict):
        return {"eligible": False, "reason": "missing Worker strategy evidence"}
    invocations = strategy.get("invocations")
    if not isinstance(invocations, list) or not 1 <= len(invocations) <= 2:
        return {"eligible": False, "reason": "Worker invocation evidence outside strategy bound"}
    if result.get("status") != "AWAITING_ACCEPTANCE" or strategy.get("validation", {}).get("status") != "valid":
        return {"eligible": False, "reason": "Worker output did not pass structural validation"}
    output = result.get("output")
    if not isinstance(output, str):
        return {"eligible": False, "reason": "Worker output missing"}
    try:
        parsed = json.loads(output)
        answer = schema_validator.model_validate(parsed)
    except Exception as exc:
        return {"eligible": False, "reason": f"Stephanie validation failed: {type(exc).__name__}: {exc}"}
    invalid_ids = sorted(set(answer.evidence_ids) - allowed_evidence_ids)
    if not answer.evidence_ids:
        return {"eligible": False, "reason": "hypothesis must cite at least one supplied evidence item"}
    if invalid_ids:
        return {"eligible": False, "reason": "hypothesis cites evidence not supplied to this task",
                "invalid_evidence_ids": invalid_ids}
    return {"eligible": True, "authority": "hypothesis_only",
            "answer": answer.model_dump(mode="json"),
            "worker_request_id": result.get("request_id"),
            "invocation_ids": [row.get("invocation_id") for row in invocations],
            "strategy_digest": strategy.get("strategy", {}).get("digest"),
            "normalization": strategy.get("validation", {}).get("normalization"),
            "correction": strategy.get("correction"),
            "resources": strategy.get("resources")}
