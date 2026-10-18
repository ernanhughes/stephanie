"""Bounded, feature-gated bridge from SIS conversations to local Worker."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import requests
from omegaconf import OmegaConf

from stephanie.data.plan_trace import ExecutionStep, PlanTrace
from stephanie.utils.secret_utils import redact_secrets


_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_WORKER_SCHEMA = "worker.work-result.v1"
_WORKER_ID = re.compile(r"^wrq_[0-9a-f]{32}$")
_WORKER_STATUSES = {
    "ACCEPTED",
    "REJECTED",
    "BLOCKED",
    "AWAITING_ACCEPTANCE",
    "FAILED",
}
_MAX_RESULT_CHARS = 12_000


class WorkerExecutionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class WorkerClient:
    """Small HTTP client for Worker's versioned loopback HTTP/JSON API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: int = 600,
    ):
        parsed = urlsplit(str(base_url))
        try:
            explicit_port = parsed.port
        except ValueError as exc:
            raise WorkerExecutionError("WORKER_URL_INVALID", "Worker URL port is invalid") from exc
        if (
            parsed.scheme != "http"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.hostname not in {"127.0.0.1", "::1"}
            or explicit_port is None
            or not 1 <= explicit_port <= 65535
        ):
            raise WorkerExecutionError(
                "WORKER_URL_INVALID",
                "Worker URL must be an explicitly configured loopback HTTP endpoint",
            )
        if not 1 <= int(timeout_seconds) <= 660:
            raise WorkerExecutionError("WORKER_CONFIG_INVALID", "Worker timeout must be 1..660 seconds")
        self.base_url = f"http://{parsed.netloc}"
        self.port = explicit_port
        self.timeout_seconds = int(timeout_seconds)

    def _json(self, method: str, path: str, *, body=None, params=None, timeout_seconds: int | None = None):
        url = self.base_url + path
        session = requests.Session()
        session.trust_env = False
        try:
            response = session.request(
                method,
                url,
                json=body,
                params=params,
                timeout=(3, self.timeout_seconds + 30) if method == "POST" else (3, timeout_seconds or 3),
                allow_redirects=False,
                stream=True,
                headers={"Accept": "application/json"},
            )
            try:
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_content(64 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > 2 * 1024 * 1024:
                        raise WorkerExecutionError("WORKER_RESPONSE_TOO_LARGE", "Worker response exceeded 2 MiB")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                if 300 <= response.status_code < 400:
                    raise WorkerExecutionError("WORKER_REDIRECT_REJECTED", "Worker API redirects are not accepted")
                if not response.ok:
                    raise WorkerExecutionError(
                        f"WORKER_HTTP_{response.status_code}",
                        f"Worker returned HTTP {response.status_code}",
                    )
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise WorkerExecutionError("WORKER_RESPONSE_INVALID", "Worker returned invalid JSON") from exc
                if not isinstance(value, dict):
                    raise WorkerExecutionError("WORKER_RESPONSE_INVALID", "Worker returned an invalid response object")
                return value
            finally:
                response.close()
        except requests.Timeout as exc:
            raise WorkerExecutionError("WORKER_TIMEOUT", "Worker request timed out; remote outcome may be unknown") from exc
        except requests.RequestException as exc:
            raise WorkerExecutionError("WORKER_UNAVAILABLE", "Could not connect to the configured Worker") from exc
        finally:
            session.close()

    def health(self) -> dict:
        body = self._json("GET", "/health")
        self._validate_health_contract(body)
        if any(not isinstance(row.get("available"), bool) for row in body["executors"]):
            raise WorkerExecutionError("WORKER_PROTOCOL_INVALID", "Worker executor health state is invalid")
        return body

    def transport_health(self) -> dict:
        """Confirm loopback listener and executor registration without probing OpenCode."""
        body = self._json("GET", "/health/transport")
        self._validate_health_contract(body)
        if any(not isinstance(row.get("id"), str) for row in body["executors"]):
            raise WorkerExecutionError("WORKER_PROTOCOL_INVALID", "Worker transport executor list is invalid")
        return body

    def _validate_health_contract(self, body: dict):
        if not isinstance(body.get("executors"), list) or not isinstance(body.get("version"), str):
            raise WorkerExecutionError("WORKER_PROTOCOL_INVALID", "Health response is not a supported Worker contract")
        if any(not isinstance(row, dict) for row in body["executors"]):
            raise WorkerExecutionError("WORKER_PROTOCOL_INVALID", "Worker executor list is invalid")
        listener = body.get("listener")
        if (
            not isinstance(listener, dict)
            or listener.get("host") not in {"127.0.0.1", "::1"}
            or listener.get("port") != self.port
        ):
            raise WorkerExecutionError("WORKER_BIND_UNVERIFIED", "Worker health did not confirm the configured loopback listener")
        security = body.get("security")
        if isinstance(security, dict) and security.get("api_auth_required") is True:
            raise WorkerExecutionError("WORKER_AUTH_UNEXPECTED", "Worker API auth is outside the configured loopback-bridge contract")

    def discover_models(self) -> list[str]:
        body = self._json("GET", "/v1/models/cli", timeout_seconds=15)
        rows = body.get("models")
        if not isinstance(rows, list):
            raise WorkerExecutionError("WORKER_MODEL_DISCOVERY_INVALID", "Worker model discovery response is invalid")
        return [str(row["id"]) for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)]

    def run(self, work_request: dict) -> dict:
        body = self._json("POST", "/v1/work", body=work_request)
        return validate_work_result(body, expected_backend=work_request["execution"]["backend"])

    def jobs(self) -> list[dict]:
        body = self._json("GET", "/v1/jobs", timeout_seconds=3)
        jobs = body.get("jobs")
        if not isinstance(jobs, list):
            raise WorkerExecutionError("WORKER_JOBS_INVALID", "Worker jobs response is invalid")
        return [job for job in jobs if isinstance(job, dict)]

    def result(self, request_id: str, *, backend: str) -> dict | None:
        if not _WORKER_ID.fullmatch(request_id or ""):
            raise WorkerExecutionError("WORKER_REQUEST_ID_INVALID", "Worker request ID has an invalid format")
        try:
            body = self._json("GET", f"/v1/work/{request_id}")
        except WorkerExecutionError as exc:
            if exc.code == "WORKER_HTTP_404":
                return None
            raise
        return validate_work_result(body, expected_backend=backend)


def validate_work_result(body: dict, *, expected_backend: str) -> dict:
    if (
        body.get("schema") != _WORKER_SCHEMA
        or not isinstance(body.get("request_id"), str)
        or not _WORKER_ID.fullmatch(body.get("request_id", ""))
        or body.get("status") not in _WORKER_STATUSES
        or not isinstance(body.get("execution"), dict)
        or body["execution"].get("backend") != expected_backend
    ):
        raise WorkerExecutionError("WORKER_RESULT_INVALID", "Worker result did not match worker.work-result.v1")
    provenance = body.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise WorkerExecutionError("WORKER_RESULT_INVALID", "Worker provenance must be an object")
    return body


class WorkerExecutionService:
    def __init__(self, *, memory, cfg, logger=None, client_factory=WorkerClient):
        self.memory = memory
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.client_factory = client_factory
        self._guard = threading.RLock()
        self._futures: dict[str, Future] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._closing = False

    @property
    def name(self):
        return "sis-worker-execution"

    def initialize(self, **_kwargs):
        # Deliberately do not connect to Worker or start any process at SIS boot.
        return None

    def health_check(self):
        return {"status": "configured" if self._settings().get("enabled", False) else "disabled"}

    def _settings(self):
        value = self.cfg.get("worker", {}) or {}
        return _plain(value)

    def _client(self) -> WorkerClient:
        settings = self._settings()
        try:
            return self.client_factory(
                settings.get("base_url", "http://127.0.0.1:7317"),
                timeout_seconds=int(settings.get("timeout_seconds", 600)),
            )
        except (TypeError, ValueError) as exc:
            raise WorkerExecutionError("WORKER_CONFIG_INVALID", "Worker configuration is invalid") from exc

    def _workspaces(self) -> dict:
        value = self._settings().get("workspaces", {}) or {}
        return dict(_plain(value))

    def _resolve_workspace(self, workspace_id: str, *, dry_run: bool = False) -> tuple[Path, dict]:
        item = self._workspaces().get(workspace_id)
        if not isinstance(item, dict):
            raise WorkerExecutionError("WORKSPACE_NOT_AUTHORIZED", "Workspace is not registered for SIS execution")
        root = item.get("root")
        project_id = item.get("project_id")
        if not isinstance(root, str) or not root.strip() or not isinstance(project_id, str) or not project_id.strip():
            raise WorkerExecutionError("WORKSPACE_CONFIG_INVALID", "Registered workspace configuration is incomplete")
        try:
            path = Path(root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkerExecutionError("WORKSPACE_NOT_AUTHORIZED", "Registered workspace path is unavailable") from exc
        if not path.is_dir():
            raise WorkerExecutionError("WORKSPACE_NOT_AUTHORIZED", "Registered workspace is not a directory")
        # This is an explicit operator verification gate, not a claim that SIS,
        # Worker, or OpenCode independently provides an OS-level sandbox.
        if dry_run:
            if item.get("dry_run_only") is not True and item.get("execution_boundary_verified") is not True:
                raise WorkerExecutionError("WORKSPACE_POLICY_UNVERIFIED", "Dry-run requires a registered dry-run-only or externally verified workspace")
        elif item.get("execution_boundary_verified") is not True:
            raise WorkerExecutionError("WORKSPACE_POLICY_UNVERIFIED", "Workspace execution boundary has not been operator-verified")
        return path, item

    def readiness(self, backend: str | None = None, *, dry_run: bool = False) -> dict:
        settings = self._settings()
        enabled = bool(settings.get("enabled", False))
        dry_run_only = bool(settings.get("dry_run_only", False))
        execution_enabled = enabled and not dry_run_only
        dry_run_enabled = enabled or dry_run_only
        backend = backend or settings.get("default_backend", "opencode")
        allowed = list(settings.get("allowed_backends", []) or [])
        model = settings.get("model")
        workspaces = []
        for workspace_id, item in self._workspaces().items():
            if isinstance(item, dict):
                workspaces.append({
                    "id": str(workspace_id),
                    "label": str(item.get("label") or workspace_id),
                    "project_id": item.get("project_id"),
                    "execution_boundary_verified": item.get("execution_boundary_verified") is True,
                    "dry_run_only": item.get("dry_run_only") is True,
                })
        dry_run_workspace_available = any(
            item["dry_run_only"] or item["execution_boundary_verified"] for item in workspaces
        )
        execution_workspace_available = any(item["execution_boundary_verified"] for item in workspaces)
        result = {
            "configured": execution_enabled or dry_run_only,
            "enabled": execution_enabled,
            "dry_run_only": dry_run_only,
            "permission_class": (
                "workspace_write"
                if backend in list(settings.get("workspace_write_backends", []) or [])
                else "inspect"
            ),
            "base_url": settings.get("base_url", "http://127.0.0.1:7317"),
            "backend": backend,
            "allowed_backends": allowed,
            "model": model,
            "workspaces": workspaces,
            "available": False,
            "dry_run_available": False,
            "message": "Worker execution is disabled by configuration.",
        }
        if (dry_run and not dry_run_enabled) or (not dry_run and not execution_enabled):
            return result
        if backend not in allowed:
            result["message"] = "The configured default backend is not explicitly allowed."
            return result
        if not isinstance(model, str) or not model.strip():
            result["message"] = "Configure an explicit Worker/OpenCode model route before enabling execution."
            return result
        if not (dry_run_workspace_available if dry_run else execution_workspace_available):
            result["message"] = (
                "Register a dry-run-only or externally verified workspace."
                if dry_run
                else "Register a disposable workspace whose filesystem, process, network, and secret boundaries have been operator-verified."
            )
            return result
        try:
            client = self._client()
            if dry_run:
                transport = client.transport_health()
                executor = next((row for row in transport["executors"] if row.get("id") == backend), None)
                if executor is None:
                    result["message"] = "Worker transport is ready, but the selected backend is not registered."
                    result["worker_version"] = transport.get("version")
                    result["error_code"] = "WORKER_EXECUTOR_NOT_REGISTERED"
                    return result
                result.update({
                    "enabled": False,
                    "dry_run_available": True,
                    "worker_version": transport["version"],
                    "executor": {"id": backend, "registered": True},
                    "model_verified": bool((settings.get("readiness", {}) or {}).get("model_verified", False)),
                    "message": "Loopback Worker transport and backend registration are ready for a dry-run; OpenCode will not be probed or invoked.",
                })
                return result
            health = client.health()
            executor = next((row for row in health["executors"] if row.get("id") == backend), None)
            if executor is None or executor.get("available") is not True:
                result["message"] = "Worker is reachable, but the selected executor is unavailable."
                result["worker_version"] = health.get("version")
                result["executor"] = executor
                return result
            models = client.discover_models()
            if model not in models:
                result["message"] = "The explicit model route was not discovered by Worker."
                result["worker_version"] = health["version"]
                result["executor"] = {"id": backend, "available": True}
                result["model_discovered"] = False
                return result
            if not (settings.get("readiness", {}) or {}).get("model_verified", False):
                result.update({
                    "worker_version": health["version"],
                    "executor": {"id": backend, "available": True, "details": executor.get("details", {})},
                    "model_discovered": True,
                    "model_verified": False,
                    "message": "The explicit model route is listed but has not passed an operator-run real inference verification; authorization/quota remain unknown.",
                    "error_code": "WORKER_MODEL_UNVERIFIED",
                })
                return result
            result.update({
                "available": execution_workspace_available,
                "dry_run_available": dry_run_workspace_available,
                "enabled": execution_workspace_available or dry_run_workspace_available,
                "worker_version": health["version"],
                "executor": {"id": backend, "available": True, "details": executor.get("details", {})},
                "model_discovered": True,
                "model_verified": True,
                "message": (
                    "Worker/executor and model route are available for a dry-run; coding execution still requires a verified isolation boundary."
                    if dry_run_workspace_available and not execution_workspace_available
                    else "Worker/executor is available and the explicit model route was discovered; provider authorization and quota are not verified."
                ),
            })
            return result
        except WorkerExecutionError as exc:
            result["message"] = str(exc)
            result["error_code"] = exc.code
            return result

    def submit(
        self,
        *,
        conversation_id: int,
        username: str,
        objective: str,
        workspace_id: str,
        backend: str,
        permission_class: str,
        submission_key: str,
        dry_run: bool = False,
        experience_context: dict | None = None,
    ) -> dict:
        if not _KEY.fullmatch(submission_key or ""):
            raise WorkerExecutionError("IDEMPOTENCY_KEY_INVALID", "Idempotency-Key must contain 1..128 safe characters")
        task = objective.strip()
        if not task or len(task) > 12_000:
            raise WorkerExecutionError("OBJECTIVE_INVALID", "Objective must contain 1..12000 characters")
        if _bounded_output(task, 12_000) != task:
            raise WorkerExecutionError("OBJECTIVE_CONTAINS_SECRET", "Remove credential-like values before submitting a Worker task")
        settings = self._settings()
        dry_run_only = bool(settings.get("dry_run_only", False))
        if (dry_run_only and not dry_run) or (
            not settings.get("enabled", False) and not (dry_run and dry_run_only)
        ):
            raise WorkerExecutionError("EXECUTION_DISABLED", "Worker execution is disabled by configuration")
        allowed_permission_classes = list(settings.get("allowed_permission_classes", ["inspect"]) or [])
        if permission_class not in allowed_permission_classes:
            raise WorkerExecutionError("PERMISSION_CLASS_NOT_ALLOWED", "The requested permission class is not enabled")
        if (
            permission_class == "workspace_write"
            and backend not in list(settings.get("workspace_write_backends", []) or [])
        ):
            raise WorkerExecutionError("PERMISSION_CLASS_NOT_ALLOWED", "Workspace writes are only enabled for the explicitly configured Docker executor")
        if backend not in list(settings.get("allowed_backends", []) or []):
            raise WorkerExecutionError("BACKEND_NOT_ALLOWED", "Selected backend is not allowed by configuration")
        experience_refs: list[dict] = []
        worker_objective = task
        if experience_context is not None:
            learning_settings = self.cfg.get("experience_learning", {}) or {}
            retrieval_settings = learning_settings.get("retrieval", {}) or {}
            if dry_run or not learning_settings.get("enabled", False) or not retrieval_settings.get("enabled", False):
                raise WorkerExecutionError("EXPERIENCE_REUSE_DISABLED", "Approved experience reuse is disabled for this execution")
            if backend != "opencode-docker":
                raise WorkerExecutionError("EXPERIENCE_BACKEND_NOT_ISOLATED", "Experience context is allowed only for the explicitly configured Docker executor")
            if not isinstance(experience_context, dict):
                raise WorkerExecutionError("EXPERIENCE_CONTEXT_INVALID", "Experience retrieval context is invalid")
            context_text = experience_context.get("context")
            refs = experience_context.get("results")
            selected = experience_context.get("selected_experience_ids")
            max_context_chars = min(8000, max(500, int(retrieval_settings.get("max_context_chars", 2500))))
            if (
                not isinstance(context_text, str)
                or not context_text.strip()
                or len(context_text) > max_context_chars
                or _bounded_output(context_text, max_context_chars) != context_text
                or not isinstance(refs, list)
                or not isinstance(selected, list)
                or not selected
                or len(selected) > max(1, int(retrieval_settings.get("max_results", 3)))
                or [item.get("experience_id") for item in refs if isinstance(item, dict)] != selected
            ):
                raise WorkerExecutionError("EXPERIENCE_CONTEXT_INVALID", "Experience context must be bounded and carry matching approved source identities")
            experience_refs = [
                {
                    "experience_id": _bounded_output(item["experience_id"], 80),
                    "memcube_id": _bounded_output(item.get("memcube_id"), 200),
                    "score": float(item.get("score", 0.0)),
                    "source": _plain(item.get("source") or {}),
                }
                for item in refs if isinstance(item, dict)
            ]
            worker_objective = f"{task}\n\n{context_text}"
            if len(worker_objective) > 16_000:
                raise WorkerExecutionError("EXPERIENCE_CONTEXT_TOO_LARGE", "Task plus approved experience exceeds the execution prompt bound")
        with self._guard:
            prior = self.memory.chats.get_external_execution_by_submission_key(conversation_id, submission_key)
            if prior is not None:
                if (
                    prior.get("objective") != task
                    or prior.get("workspace_id") != workspace_id
                    or prior.get("backend") != backend
                    or prior.get("permission_class") != permission_class
                    or bool(prior.get("dry_run")) != bool(dry_run)
                    or prior.get("experience_ids", []) != [item["experience_id"] for item in experience_refs]
                ):
                    raise WorkerExecutionError("IDEMPOTENCY_CONFLICT", "Idempotency-Key was already used for another execution")
                return _public_record(prior)
        path, workspace = self._resolve_workspace(workspace_id, dry_run=dry_run)
        readiness = self.readiness(backend, dry_run=dry_run)
        if not readiness.get("dry_run_available" if dry_run else "available"):
            raise WorkerExecutionError(readiness.get("error_code", "WORKER_NOT_READY"), readiness["message"])
        model = str(settings["model"])
        request_fingerprint = hashlib.sha256(json.dumps(
            {"objective": task, "workspace_id": workspace_id, "backend": backend, "model": model, "permission_class": permission_class, "dry_run": bool(dry_run), "experience_ids": [item["experience_id"] for item in experience_refs]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()).hexdigest()
        execution_id = f"sex_{uuid4().hex}"
        now = _now()
        record = {
            "execution_id": execution_id,
            "conversation_id": conversation_id,
            "originating_message_id": None,
            "originating_turn_id": None,
            "username": username,
            "project_id": workspace["project_id"],
            "workspace_id": workspace_id,
            "backend": backend,
            "model": None,
            "requested_model": model,
            "permission_class": permission_class,
            "dry_run": bool(dry_run),
            "objective": task,
            "worker_objective": worker_objective,
            "experience_ids": [item["experience_id"] for item in experience_refs],
            "experience_retrieval": experience_refs,
            "experience_context": (
                _bounded_output(experience_context.get("context"), 8000)
                if experience_context is not None else None
            ),
            "request_fingerprint": request_fingerprint,
            "status": "QUEUED",
            "trace_id": f"worker:{execution_id}",
            "worker_request_id": None,
            "opencode_session_id": None,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "result_summary": None,
            "error_category": None,
            "duration_ms": None,
            "artifacts": [],
            "provenance": {},
        }
        with self._guard:
            if self._closing:
                raise WorkerExecutionError("SIS_SHUTTING_DOWN", "SIS is shutting down and is not accepting executions")
            prior = self.memory.chats.get_external_execution_by_submission_key(conversation_id, submission_key)
            if prior is not None:
                if (
                    prior.get("objective") != task
                    or prior.get("workspace_id") != workspace_id
                    or prior.get("backend") != backend
                    or prior.get("permission_class") != permission_class
                    or bool(prior.get("dry_run")) != bool(dry_run)
                    or prior.get("experience_ids", []) != [item["experience_id"] for item in experience_refs]
                ):
                    raise WorkerExecutionError("IDEMPOTENCY_CONFLICT", "Idempotency-Key was already used for another execution")
                return _public_record(prior)
            max_pending = max(1, int(settings.get("max_concurrent", 2))) + max(0, int(settings.get("max_queued", 2)))
            if len(self._futures) >= max_pending:
                raise WorkerExecutionError("EXECUTION_CAPACITY", "SIS has reached its bounded execution capacity")
            message, stored, created = self.memory.chats.start_external_execution(
                conversation_id,
                submission_key,
                request_fingerprint,
                task,
                record,
                max_records=int(settings.get("max_records_per_conversation", 100)),
            )
            if not created:
                return _public_record(stored)
            stored = self.memory.chats.update_external_execution(
                conversation_id,
                execution_id,
                {"originating_message_id": message.id},
            )
            try:
                self._write_trace(stored, initial=True)
            except Exception as exc:
                blocked = self.memory.chats.update_external_execution(
                    conversation_id,
                    execution_id,
                    {"status": "BLOCKED", "error_category": "PLANTRACE_PERSISTENCE_FAILED", "completed_at": _now()},
                )
                self._log("SISWorkerPlanTraceWriteFailed", blocked, exc)
                return _public_record(blocked)
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=max(1, int(settings.get("max_concurrent", 2))),
                    thread_name_prefix="sis-worker",
                )
            future = self._pool.submit(self._dispatch, conversation_id, execution_id, path, stored)
            self._futures[execution_id] = future
            future.add_done_callback(lambda _future, key=execution_id: self._forget(key))
        return _public_record(stored)

    def _forget(self, execution_id: str):
        with self._guard:
            self._futures.pop(execution_id, None)

    def _dispatch(self, conversation_id: int, execution_id: str, workspace_root: Path, record: dict):
        if self._closing:
            return
        started = time.monotonic()
        started_at = _now()
        try:
            self._save(conversation_id, execution_id, {"status": "RUNNING", "started_at": started_at})
            self._write_trace(self.memory.chats.get_external_execution(conversation_id, execution_id))
        except Exception as exc:
            self._save(conversation_id, execution_id, {
                "status": "BLOCKED",
                "error_category": "PLANTRACE_UPDATE_FAILED",
                "error_message": "Execution evidence could not be updated before Worker dispatch.",
                "completed_at": _now(),
            })
            self._log("SISWorkerPlanTraceWriteFailed", record, exc)
            return
        if self._closing:
            return
        body = {
            "schema": "worker.work-request.v1",
            "app": "stephanie",
            "objective": record.get("worker_objective") or record["objective"],
            "workspace": {"root": str(workspace_root), "project_id": record["project_id"]},
            "execution": {"backend": record["backend"], "model": record["requested_model"]},
            "options": {"acceptance": "human", "dry_run": bool(record.get("dry_run"))},
            "metadata": {
                "source": "sis",
                "operation_id": execution_id,
                "stage_id": "OPENCODE",
                "experience_ids": record.get("experience_ids", []),
            },
        }
        try:
            result = self._client().run(body)
        except WorkerExecutionError as exc:
            if self._closing:
                return
            remote = self._find_remote_job(execution_id, record["backend"])
            if remote and remote[0]:
                worker_id, result = remote
                if result is None:
                    changes = {
                        "status": "RUNNING",
                        "worker_request_id": worker_id,
                        "error_category": "SUBMISSION_RESPONSE_LOST",
                        "duration_ms": int((time.monotonic() - started) * 1000),
                    }
                    self._save(conversation_id, execution_id, changes)
                    self._write_trace(self.memory.chats.get_external_execution(conversation_id, execution_id))
                    return
            else:
                status = "BLOCKED" if _known_pre_execution_rejection(exc.code) else "INTERRUPTED"
                changes = {
                    "status": status,
                    "error_category": "OUTCOME_UNKNOWN" if status == "INTERRUPTED" else exc.code,
                    "error_message": str(exc)[:240],
                    "completed_at": _now(),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
                self._save(conversation_id, execution_id, changes)
                self._write_trace(self.memory.chats.get_external_execution(conversation_id, execution_id))
                self._log("SISWorkerDispatchUncertain", self.memory.chats.get_external_execution(conversation_id, execution_id), exc)
                return
        except Exception as exc:
            if self._closing:
                return
            changes = {
                "status": "FAILED",
                "error_category": "CLIENT_ERROR",
                "error_message": "Worker request failed before a result could be validated.",
                "completed_at": _now(),
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
            self._save(conversation_id, execution_id, changes)
            self._write_trace(self.memory.chats.get_external_execution(conversation_id, execution_id))
            self._log("SISWorkerDispatchFailed", self.memory.chats.get_external_execution(conversation_id, execution_id), exc)
            return

        if not self._closing:
            self._apply_result(conversation_id, execution_id, result, int((time.monotonic() - started) * 1000))

    def _find_remote_job(
        self,
        execution_id: str,
        backend: str,
        *,
        jobs: list[dict] | None = None,
        worker_request_id: str | None = None,
    ) -> tuple[str, dict | None] | None:
        try:
            if worker_request_id:
                result = self._client().result(worker_request_id, backend=backend)
                if result is not None:
                    return worker_request_id, result
            for job in self._client().jobs() if jobs is None else jobs:
                if job.get("operation_id") != execution_id or not isinstance(job.get("job_id"), str):
                    continue
                worker_id = job["job_id"]
                raw = job.get("result")
                if isinstance(raw, dict):
                    validated = validate_work_result(raw, expected_backend=backend)
                    if validated["request_id"] != worker_id:
                        return None
                    return worker_id, validated
                if str(job.get("status", "")).upper() != "RUNNING":
                    result = self._client().result(worker_id, backend=backend)
                    if result is not None:
                        return worker_id, result
                return worker_id, None
        except WorkerExecutionError:
            return None
        return "", None  # Worker responded, but no retained job matches this execution.

    def _apply_result(self, conversation_id: int, execution_id: str, result: dict, duration_ms: int | None):
        status = {
            "ACCEPTED": "ACCEPTED" if isinstance(result.get("acceptance_id"), str) and result["acceptance_id"] else "COMPLETED_UNVERIFIED",
            "REJECTED": "REJECTED",
            "BLOCKED": "BLOCKED",
            "FAILED": "FAILED",
            "AWAITING_ACCEPTANCE": "COMPLETED_UNVERIFIED",
        }[result["status"]]
        provenance = result.get("provenance") or {}
        execution = result.get("execution") or {}
        prior_record = self.memory.chats.get_external_execution(conversation_id, execution_id) or {}
        summary = _bounded_output(result.get("output"))
        worker_message = _bounded_output(result.get("message"), 1000)
        error_category = _error_category(result)
        if prior_record.get("dry_run"):
            dry_run_confirmed = (
                result.get("status") == "AWAITING_ACCEPTANCE"
                and worker_message == "dry run: OpenCode was not invoked"
                and not summary
                and not provenance.get("session_id")
            )
            if dry_run_confirmed:
                status = "DRY_RUN_NO_AGENT_EXECUTION"
                error_category = None
            else:
                status = "FAILED"
                error_category = "DRY_RUN_CONTRACT_UNVERIFIED"
        changes = {
            "status": status,
            "worker_request_id": result["request_id"],
            "backend": execution.get("backend"),
            "requested_model": execution.get("model") or prior_record.get("requested_model"),
            "reported_model_route": execution.get("model"),
            "model": provenance.get("model") if isinstance(provenance.get("model"), str) else None,
            "provider": provenance.get("provider") if isinstance(provenance.get("provider"), str) else None,
            "opencode_session_id": provenance.get("session_id"),
            "container_id": _safe_identifier(provenance.get("container_id")),
            "model_relay_container_id": _safe_identifier(provenance.get("model_relay_container_id")),
            "image_digest": _safe_identifier(provenance.get("image_digest")),
            "provider_request_model": _safe_identifier(provenance.get("provider_request_model")),
            "provider_response_status": _safe_identifier(provenance.get("provider_response_status")),
            "verification_command": _safe_identifier(provenance.get("verification_command")),
            "verification_exit_code": _safe_identifier(provenance.get("verification_exit_code")),
            "verification_status": _safe_identifier(provenance.get("verification_status")),
            "worker_work_id": _safe_identifier(result.get("work_id")),
            "worker_run_id": _safe_identifier(result.get("run_id")),
            "acceptance_id": _safe_identifier(result.get("acceptance_id")),
            "result_summary": summary,
            "worker_message": worker_message,
            "unresolved": _safe_unresolved(result.get("unresolved")),
            "artifacts": _safe_artifacts(result.get("artifacts")),
            "provenance": _safe_provenance(provenance),
            "completed_at": _now(),
            "duration_ms": duration_ms,
            "error_category": (
                "ACCEPTANCE_ID_MISSING"
                if result.get("status") == "ACCEPTED" and status != "ACCEPTED"
                else error_category
            ),
        }
        writes_assistant = bool(summary and status in {"COMPLETED_UNVERIFIED", "ACCEPTED", "REJECTED"})
        stored = self._save(
            conversation_id,
            execution_id,
            {
                **changes,
                **({"status": "RUNNING", "completed_at": None} if writes_assistant else {}),
            },
        )
        if writes_assistant:
            user_message_id = stored.get("originating_message_id")
            request_key = f"stn03-{execution_id}"
            assistant_meta = {
                "runtime": "worker",
                "worker_execution_id": execution_id,
                "worker_request_id": result["request_id"],
                "backend": execution.get("backend"),
                "model": provenance.get("model"),
                "requested_model": execution.get("model"),
                "status": status,
                "plan_trace_id": stored.get("trace_id"),
                "experience_ids": stored.get("experience_ids", []),
            }
            try:
                assistant, turn = self.memory.chats.complete_assistant_turn(
                    conversation_id,
                    user_message_id,
                    request_key,
                    summary,
                    meta=redact_secrets(assistant_meta),
                )
                stored = self._save(conversation_id, execution_id, {
                    "assistant_message_id": assistant.id,
                    "originating_turn_id": turn.id,
                    "status": status,
                    "completed_at": changes["completed_at"],
                })
            except Exception as exc:
                stored = self._save(conversation_id, execution_id, {
                    "status": status,
                    "completed_at": changes["completed_at"],
                    "conversation_link_error": type(exc).__name__,
                })
                self._log("SISWorkerConversationLinkFailed", stored, exc)
        try:
            self._write_trace(stored)
        except Exception as exc:
            stored = self._save(conversation_id, execution_id, {
                "plan_trace_update_error": type(exc).__name__,
            })
            self._log("SISWorkerPlanTraceWriteFailed", stored, exc)

    def _save(self, conversation_id: int, execution_id: str, changes: dict) -> dict:
        safe_changes = redact_secrets(changes)
        safe_changes.pop("username", None)
        safe_changes["updated_at"] = _now()
        stored = self.memory.chats.update_external_execution(conversation_id, execution_id, safe_changes)
        if stored is None:
            raise WorkerExecutionError("EXECUTION_EVIDENCE_MISSING", "Persisted execution evidence was not found")
        return stored

    def _write_trace(self, record: dict | None, *, initial: bool = False):
        if not record:
            raise WorkerExecutionError("EXECUTION_EVIDENCE_MISSING", "Execution evidence was not found")
        status = record.get("status", "QUEUED")
        output = _bounded_output(record.get("result_summary") or record.get("error_message") or status, 2000)
        steps = [
            ExecutionStep(
                step_id="worker_submission",
                step_type="external_execution",
                description="Submitted an explicitly selected Worker backend for a registered SIS workspace.",
                input_text=record.get("objective", "")[:2000],
                output_text=f"execution_id={record['execution_id']} status={status}",
                scores=None,
                agent_name="worker",
                agent_role="executor",
                step_order=1,
                meta={
                    "worker_request_id": record.get("worker_request_id"),
                    "container_id": record.get("container_id"),
                    "model_relay_container_id": record.get("model_relay_container_id"),
                    "image_digest": record.get("image_digest"),
                    "provider_request_model": record.get("provider_request_model"),
                    "provider_response_status": record.get("provider_response_status"),
                    "verification_command": record.get("verification_command"),
                    "verification_exit_code": record.get("verification_exit_code"),
                    "verification_status": record.get("verification_status"),
                    "experience_ids": record.get("experience_ids", []),
                    "experience_retrieval": record.get("experience_retrieval", []),
                    "backend": record.get("backend"),
                    "model": record.get("model"),
                    "requested_model": record.get("requested_model"),
                    "status": status,
                    "duration_ms": record.get("duration_ms"),
                },
            )
        ]
        if not initial and record.get("started_at") and not record.get("completed_at"):
            steps.append(ExecutionStep(
                step_id="worker_dispatch_started",
                step_type="action",
                description="SIS dispatch began; Worker acceptance/completion is not yet confirmed.",
                output_text=f"execution_id={record['execution_id']} status={status}",
                scores=None,
                agent_name="worker",
                agent_role="executor",
                step_order=2,
                meta={
                    "worker_request_id": record.get("worker_request_id"),
                    "status": status,
                },
            ))
        elif not initial and record.get("completed_at"):
            steps.append(ExecutionStep(
                step_id="worker_result",
                step_type="observation",
                description="Observed the Worker execution outcome; acceptance is recorded only when Worker supplies it.",
                output_text=output,
                scores=None,
                agent_name="worker",
                agent_role="observation",
                step_order=2,
                meta={
                    "worker_request_id": record.get("worker_request_id"),
                    "opencode_session_id": record.get("opencode_session_id"),
                    "container_id": record.get("container_id"),
                    "model_relay_container_id": record.get("model_relay_container_id"),
                    "image_digest": record.get("image_digest"),
                    "provider_request_model": record.get("provider_request_model"),
                    "provider_response_status": record.get("provider_response_status"),
                    "verification_command": record.get("verification_command"),
                    "verification_exit_code": record.get("verification_exit_code"),
                    "verification_status": record.get("verification_status"),
                    "experience_ids": record.get("experience_ids", []),
                    "experience_retrieval": record.get("experience_retrieval", []),
                    "status": status,
                    "error_category": record.get("error_category"),
                },
            ))
        trace = PlanTrace(
            trace_id=record["trace_id"],
            goal_text=record.get("objective", "")[:12_000],
            goal_id=None,  # PlanTraceORM supports external traces without a Goal row.
            input_data={
                "execution_id": record["execution_id"],
                "conversation_id": record["conversation_id"],
                "user_message_id": record.get("originating_message_id"),
                "project_id": record.get("project_id"),
                "workspace_id": record.get("workspace_id"),
                "dry_run": bool(record.get("dry_run")),
                "experience_ids": record.get("experience_ids", []),
            },
            plan_signature="worker.work-request.v1",
            execution_steps=steps,
            final_output_text=output,
            status=status.lower(),
            created_at=record.get("created_at", _now()),
            meta={
                "source": "sis_worker",
                "execution_id": record["execution_id"],
                "conversation_id": record["conversation_id"],
                "user_message_id": record.get("originating_message_id"),
                "worker_request_id": record.get("worker_request_id"),
                "backend": record.get("backend"),
                "model": record.get("model"),
                "requested_model": record.get("requested_model"),
                "container_id": record.get("container_id"),
                "model_relay_container_id": record.get("model_relay_container_id"),
                "image_digest": record.get("image_digest"),
                "provider_request_model": record.get("provider_request_model"),
                "provider_response_status": record.get("provider_response_status"),
                "verification_command": record.get("verification_command"),
                "verification_exit_code": record.get("verification_exit_code"),
                "verification_status": record.get("verification_status"),
                "experience_ids": record.get("experience_ids", []),
                "experience_retrieval": record.get("experience_retrieval", []),
                "dry_run": bool(record.get("dry_run")),
                "status": status,
                "opencode_session_id": record.get("opencode_session_id"),
                "duration_ms": record.get("duration_ms"),
            },
        )
        store = self.memory.plan_traces
        if initial:
            store.insert(trace)
        else:
            store.upsert(trace)

    def get(self, conversation_id: int, execution_id: str) -> dict | None:
        record = self.memory.chats.get_external_execution(conversation_id, execution_id)
        if not record:
            return None
        uncertain = record.get("status") == "INTERRUPTED" and record.get("error_category") == "OUTCOME_UNKNOWN"
        if record.get("status") in {"QUEUED", "RUNNING"} or uncertain:
            remote = self._find_remote_job(
                execution_id,
                record["backend"],
                worker_request_id=record.get("worker_request_id"),
            )
            record = self._reconcile_record(conversation_id, record, remote)
        return _public_record(record)

    def list(self, conversation_id: int) -> list[dict]:
        records = self.memory.chats.list_external_executions(conversation_id)
        to_reconcile = [
            item for item in records
            if item.get("status") in {"QUEUED", "RUNNING"}
            or (item.get("status") == "INTERRUPTED" and item.get("error_category") == "OUTCOME_UNKNOWN")
        ]
        jobs = None
        if to_reconcile:
            try:
                jobs = self._client().jobs()
            except WorkerExecutionError:
                pass
        result = []
        for record in records:
            if record in to_reconcile and jobs is not None:
                remote = self._find_remote_job(
                    record["execution_id"],
                    record["backend"],
                    jobs=jobs,
                    worker_request_id=record.get("worker_request_id"),
                )
                reconciled = self._reconcile_record(conversation_id, record, remote)
                result.append(_public_record(reconciled))
            else:
                result.append(_public_record(record))
        return result

    def _reconcile_record(self, conversation_id: int, record: dict, remote) -> dict:
        execution_id = record["execution_id"]
        uncertain = record.get("status") == "INTERRUPTED" and record.get("error_category") == "OUTCOME_UNKNOWN"
        if remote and remote[0]:
            worker_id, result = remote
            if result:
                if execution_id not in self._futures:
                    self._apply_result(conversation_id, execution_id, result, record.get("duration_ms"))
                    return self.memory.chats.get_external_execution(conversation_id, execution_id)
                if worker_id != record.get("worker_request_id"):
                    return self._save(conversation_id, execution_id, {"worker_request_id": worker_id})
            else:
                changes = {"worker_request_id": worker_id}
                if uncertain:
                    changes.update({"status": "RUNNING", "error_category": None, "error_message": None, "completed_at": None})
                if worker_id != record.get("worker_request_id") or uncertain:
                    return self._save(conversation_id, execution_id, changes)
        elif remote == ("", None) and execution_id not in self._futures and not uncertain:
            return self._save(conversation_id, execution_id, {
                "status": "INTERRUPTED",
                "error_category": "OUTCOME_UNKNOWN",
                "error_message": "Worker no longer retains the correlated job; do not retry automatically.",
                "completed_at": _now(),
            })
        # A Worker connection failure or missing retained job never triggers a retry.
        return record

    def shutdown(self):
        with self._guard:
            self._closing = True
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=False)
            self._pool = None

    def _log(self, event: str, record: dict | None, exc: Exception):
        payload = {
            "execution_id": (record or {}).get("execution_id"),
            "conversation_id": (record or {}).get("conversation_id"),
            "error_type": type(exc).__name__,
        }
        if isinstance(self.logger, logging.Logger):
            self.logger.warning("%s: %s", event, payload)
        else:
            self.logger.log(event, payload)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _bounded_output(value, limit: int = _MAX_RESULT_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            text = str(value)
    text = str(redact_secrets(text))
    return text[:limit]


def _safe_provenance(value: dict) -> dict:
    allowed = {"project_id", "session_id", "work_id", "worker_request_id", "application", "ledger_dir", "edge_record_ids", "provider", "model", "container_id", "model_relay_container_id", "image_digest", "provider_request_model", "provider_response_status"}
    safe = {}
    for key in allowed:
        item = value.get(key)
        if isinstance(item, str):
            safe[key] = _bounded_output(item, 500)
        elif key == "edge_record_ids" and isinstance(item, list):
            safe[key] = [_bounded_output(entry, 200) for entry in item[:50] if isinstance(entry, str)]
    return redact_secrets(safe)


def _safe_identifier(value) -> str | None:
    if not isinstance(value, str):
        return None
    safe = _bounded_output(value, 200)
    return safe if safe == value else None


def _safe_unresolved(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:20]:
        if not isinstance(item, dict):
            continue
        code = item.get("code")
        message = item.get("message")
        result.append({
            "code": _bounded_output(code, 80),
            "message": _bounded_output(message, 1000),
        })
    return result


def _safe_artifacts(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:50]:
        if not isinstance(item, dict):
            continue
        result.append({
            key: _bounded_output(item[key], 200)
            for key in ("id", "schema", "kind")
            if isinstance(item.get(key), str)
        })
    return result


def _error_category(result: dict) -> str | None:
    if result.get("status") not in {"FAILED", "BLOCKED", "REJECTED"}:
        return None
    unresolved = result.get("unresolved")
    if isinstance(unresolved, list) and unresolved and isinstance(unresolved[0], dict):
        code = unresolved[0].get("code")
        if isinstance(code, str):
            return _bounded_output(code, 80)
    return result.get("status")


def _known_pre_execution_rejection(code: str) -> bool:
    match = re.fullmatch(r"WORKER_HTTP_(\d{3})", code)
    return bool(match and 400 <= int(match.group(1)) < 500)


def _public_record(record: dict) -> dict:
    result = {key: value for key, value in record.items() if key not in {"username", "request_fingerprint"}}
    labels = {
        "DRY_RUN_NO_AGENT_EXECUTION": "DRY RUN — NO AGENT EXECUTION",
        "DRY_RUN_BLOCKED": "DRY RUN BLOCKED — NO AGENT EXECUTION",
    }
    result["status_label"] = labels.get(result.get("status"), result.get("status"))
    return result
