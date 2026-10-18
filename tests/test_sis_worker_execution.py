from __future__ import annotations

import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import JSON, create_engine
from sqlalchemy.orm import sessionmaker
import requests
from unittest.mock import patch

from sis.routes.chats import router as chats_router
from sis.security import AuthSettings, SISAuthMiddleware, SessionStore, hash_password
from sis.services.chat_context import build_chat_messages
from sis.services.chat_execution import ConversationExecutionService
from sis.services.worker_execution import (
    WorkerClient,
    WorkerExecutionError,
    WorkerExecutionService,
    validate_work_result,
)
from stephanie.services.execution_experience import ExecutionExperienceService, ExperienceError
from sis.template_utils import SISTemplates
from stephanie.memory.chat_store import ChatStore
from stephanie.memory.memory_tool import MemoryTool as _RegisterAllORMMappings
from stephanie.memory.plan_trace_store import PlanTraceStore
from stephanie.orm.base import Base
from stephanie.orm.chat import ChatConversationORM, ChatMessageORM, ChatTurnORM
from stephanie.orm.evaluation import EvaluationORM
from stephanie.orm.goal import GoalORM
from stephanie.orm.pipeline_run import PipelineRunORM
from stephanie.orm.plan_trace import ExecutionStepORM, PlanTraceORM
from stephanie.orm.prompt import PromptORM
from stephanie.orm.symbolic_rule import SymbolicRuleORM


class TraceStore:
    def __init__(self):
        self.traces = {}
        self.lock = threading.Lock()

    def insert(self, trace):
        with self.lock:
            self.traces[trace.trace_id] = trace
        return trace

    def upsert(self, trace):
        with self.lock:
            self.traces[trace.trace_id] = trace
        return trace

    def get_by_trace_id(self, trace_id):
        with self.lock:
            return self.traces.get(trace_id)


class FakeWorker:
    def __init__(self, *, status="AWAITING_ACCEPTANCE", result_output="Found a concrete issue.", available=True, post_status=200, include_acceptance=True, retain_lost_result=False, backend="opencode"):
        self.status = status
        self.result_output = result_output
        self.available = available
        self.post_status = post_status
        self.include_acceptance = include_acceptance
        self.retain_lost_result = retain_lost_result
        self.backend = backend
        self.requests = []
        self.auth_headers = []
        self.jobs = []
        self.post_count = 0
        self.health_calls = 0
        self.transport_health_calls = 0
        self.model_discovery_calls = 0

    def handler(self):
        state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _send(self, status, value):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                state.auth_headers.append(self.headers.get("Authorization"))
                if self.path == "/health":
                    state.health_calls += 1
                    self._send(200, {
                        "version": "0.2.0",
                        "executors": [{"id": state.backend, "available": state.available, "details": {"version": "opencode v2.0.29"}}],
                        "listener": {"host": "127.0.0.1", "port": state.server_port},
                        "security": {"api_auth_required": False},
                    })
                elif self.path == "/health/transport":
                    state.transport_health_calls += 1
                    self._send(200, {
                        "status": "ready",
                        "version": "0.2.0",
                        "executors": [{"id": state.backend}],
                        "listener": {"host": "127.0.0.1", "port": state.server_port},
                        "security": {"api_auth_required": False},
                    })
                elif self.path == "/v1/models/cli":
                    state.model_discovery_calls += 1
                    self._send(200, {"schema_version": "worker.cli-models.v1", "models": [{"id": "fixture/provider-model"}]})
                elif self.path == "/v1/jobs":
                    self._send(200, {"jobs": list(state.jobs)})
                elif self.path.startswith("/v1/work/"):
                    request_id = self.path.rsplit("/", 1)[-1]
                    result = next((job.get("result") for job in state.jobs if job.get("job_id") == request_id), None)
                    self._send(200, result) if result else self._send(404, {"error": {"code": "NOT_FOUND"}})
                else:
                    self._send(404, {"error": {"code": "NOT_FOUND"}})

            def do_POST(self):
                state.auth_headers.append(self.headers.get("Authorization"))
                if self.path != "/v1/work":
                    self._send(404, {"error": {"code": "NOT_FOUND"}})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                state.requests.append(body)
                state.post_count += 1
                if state.post_status != 200 and not state.retain_lost_result:
                    self._send(state.post_status, {"error": {"code": "FIXTURE_FAILURE"}})
                    return
                request_id = (
                    "wrq_0123456789abcdef0123456789abcdef"
                    if state.post_count == 1
                    else f"wrq_{state.post_count:032x}"
                )
                dry_run = body.get("options", {}).get("dry_run") is True
                result = {
                    "schema": "worker.work-result.v1",
                    "request_id": request_id,
                    "status": "AWAITING_ACCEPTANCE" if dry_run else state.status,
                    "execution": {"backend": state.backend, "model": "fixture/provider-model"},
                    "provenance": ({"project_id": "fixture-project", "worker_request_id": request_id} if dry_run else {"project_id": "fixture-project", "session_id": "ses_fixture", "worker_request_id": request_id}),
                    **({"message": "dry run: OpenCode was not invoked"} if dry_run else {"output": state.result_output}),
                    **({"acceptance_id": "accept_fixture"} if not dry_run and state.status == "ACCEPTED" and state.include_acceptance else {}),
                }
                state.jobs.append({
                    "job_id": result["request_id"],
                    "operation_id": body.get("metadata", {}).get("operation_id"),
                    "status": result["status"],
                    "result": result,
                })
                self._send(state.post_status, result if state.post_status == 200 else {"error": {"code": "RESPONSE_LOST"}})

        return Handler


class RunningWorker:
    def __init__(self, worker: FakeWorker):
        self.worker = worker
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), worker.handler())
        self.worker.server_port = self.server.server_port
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def make_database(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "sis-worker-test.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(
        engine,
        tables=[ChatConversationORM.__table__, ChatMessageORM.__table__, ChatTurnORM.__table__],
    )
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, sessions, ChatStore(sessions)


def create_sqlite_plan_trace_tables(engine):
    # PlanTraceORM's only PostgreSQL-specific column is ARRAY(String). For this
    # temporary SQLite fixture, represent that unused column as JSON; production
    # ORM types and schemas are unchanged (pytest monkeypatch restores the type).
    tables = [
        GoalORM.__table__,
        PipelineRunORM.__table__,
        PromptORM.__table__,
        SymbolicRuleORM.__table__,
        PlanTraceORM.__table__,
        EvaluationORM.__table__,
        ExecutionStepORM.__table__,
    ]
    for table in tables:
        table.create(engine, checkfirst=True)


def make_app(tmp_path, worker: FakeWorker, *, enabled=True, chat_enabled=True, owner="local"):
    engine, sessions, chats = make_database(tmp_path)
    conversation = chats.add_conversation({
        "provider": "stephanie",
        "title": "Worker integration fixture",
        "meta": {"sis_owner": owner},
    })
    traces = TraceStore()
    memory = SimpleNamespace(chats=chats, plan_traces=traces)
    worker_config = {
        "enabled": enabled,
        "base_url": "http://127.0.0.1:1",
        "timeout_seconds": 2,
        "default_backend": "opencode",
        "allowed_backends": ["opencode"],
        "model": "fixture/provider-model",
        "readiness": {"model_verified": True},  # Controlled fake-Worker readiness override.
        "max_concurrent": 2,
        "max_queued": 2,
        "workspaces": {
            "fixture": {
                "root": str(tmp_path),
                "project_id": "fixture-project",
                "label": "Disposable fixture",
                "execution_boundary_verified": True,
            }
        },
    }
    app = FastAPI()
    app.state.cfg = {
        "chat_runtime": {
            "enabled": chat_enabled,
            "model": {"name": "fixture/model", "api_base": "http://model.invalid"},
            "readiness": {"model_verified": True, "live_persistence_verified": True},
        },
        "worker": worker_config,
        "experience_learning": {
            "enabled": True,
            "model_assisted_extraction": False,
            "retrieval": {"enabled": True, "max_results": 3, "candidate_pool": 20, "max_context_chars": 1200},
        },
    }
    app.state.database_ready = True
    app.state.auth_settings = AuthSettings(
        username="local",
        password_hash=hash_password("fixture-password-not-used"),
        session_secret=b"worker-execution-test-session-secret",
        allowed_hosts=("127.0.0.1",),
        allowed_origins=("http://127.0.0.1:8000",),
    )
    app.state.auth_sessions = SessionStore(app.state.auth_settings)
    app.state.memory = memory
    app.state.chat_execution = ConversationExecutionService(memory=memory, container=None, cfg=app.state.cfg)
    app.state.worker_execution = WorkerExecutionService(memory=memory, cfg=app.state.cfg)
    app.state.execution_experience = ExecutionExperienceService(memory=memory, cfg=app.state.cfg)
    app.state.templates = SISTemplates(directory="sis/templates")
    app.add_middleware(SISAuthMiddleware)
    app.include_router(chats_router)
    app.mount("/static", StaticFiles(directory="sis/static"), name="static")
    token, session = app.state.auth_sessions.create("local")
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    client.cookies.set("sis_session", token, domain="127.0.0.1", path="/")
    client.headers.update({"Origin": "http://127.0.0.1:8000", "X-CSRF-Token": session.csrf_token})
    worker_config["base_url"] = ""
    return app, client, conversation.id, worker_config, traces, sessions, engine


def configure_worker(worker_config, url):
    worker_config["base_url"] = url


def wait_terminal(client, conversation_id, execution_id):
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        response = client.get(f"/chats/{conversation_id}/executions/{execution_id}")
        assert response.status_code == 200
        record = response.json()["execution"]
        if record["status"] not in {"QUEUED", "RUNNING"}:
            service = client.app.state.worker_execution
            deadline = time.monotonic() + 2
            while execution_id in service._futures and time.monotonic() < deadline:
                time.sleep(0.01)
            latest = client.get(f"/chats/{conversation_id}/executions/{execution_id}")
            assert latest.status_code == 200
            return latest.json()["execution"]
        time.sleep(0.05)
    raise AssertionError("Worker execution did not reach a terminal state")


class ExperienceChatFixture:
    def __init__(self, record, messages):
        self.record = deepcopy(record)
        self.messages = messages

    def get_external_execution(self, conversation_id, execution_id):
        if self.record["conversation_id"] != conversation_id or self.record["execution_id"] != execution_id:
            return None
        return deepcopy(self.record)

    def record_external_execution_review(self, conversation_id, execution_id, **review):
        if self.record["conversation_id"] != conversation_id or self.record["execution_id"] != execution_id:
            return None
        history = self.record.setdefault("human_review_history", [])
        prior = next((item for item in history if item["review_id"] == review["review_id"]), None)
        new = {key: review[key] for key in ("review_id", "reviewer", "decision", "notes", "evidence_refs", "accepted_scope", "reviewed_at")}
        if prior is not None:
            assert {k: v for k, v in prior.items() if k != "reviewed_at"} == {k: v for k, v in new.items() if k != "reviewed_at"}
            return deepcopy(self.record)
        history.append(new)
        self.record["human_review"] = new
        self.record["human_review_status"] = new["decision"]
        return deepcopy(self.record)

    def update_external_execution(self, conversation_id, execution_id, changes):
        if self.record["conversation_id"] != conversation_id or self.record["execution_id"] != execution_id:
            return None
        self.record.update(deepcopy(changes))
        return deepcopy(self.record)

    def get_messages(self, conversation_id):
        assert conversation_id == self.record["conversation_id"]
        return self.messages


class ExperienceTraceFixture:
    def __init__(self, meta):
        self.meta = meta
        self.execution_steps = [SimpleNamespace(step_id="worker_submission"), SimpleNamespace(step_id="worker_result")]


class ExperienceTraceStoreFixture:
    def __init__(self, meta):
        self.trace = ExperienceTraceFixture(meta)

    def get_by_trace_id(self, trace_id):
        return self.trace if trace_id == self.trace.meta.get("trace_id") else None


class ExperienceCaseBookStoreFixture:
    def __init__(self):
        self.candidates = {}

    def upsert_execution_experience_candidate(self, candidate, *, goal_id):
        assert candidate["task"]["goal_id"] == goal_id
        experience_id = candidate["experience_id"]
        existing = self.candidates.get(experience_id)
        if existing:
            assert existing["candidate"]["candidate_hash"] == candidate["candidate_hash"]
            return {**existing, "created": False}
        item = {"casebook_id": 51, "case_id": 91, "casebook_name": f"stn05-experience:{experience_id}", "experience_id": experience_id, "candidate": deepcopy(candidate), "created": True}
        self.candidates[experience_id] = item
        return deepcopy(item)

    def get_execution_experience_candidate(self, experience_id):
        return deepcopy(self.candidates.get(experience_id))

    def review_execution_experience_candidate(self, experience_id, review):
        item = self.candidates.get(experience_id)
        if not item:
            return None
        candidate = item["candidate"]
        history = candidate.setdefault("candidate_review_history", [])
        prior = next((row for row in history if row["review_id"] == review["review_id"]), None)
        if prior is None:
            history.append(deepcopy(review))
            candidate["candidate_review"] = deepcopy(review)
            candidate["candidate_review_status"] = {"APPROVE":"APPROVED","REJECT":"REJECTED","NEEDS_REVISION":"NEEDS_REVISION"}[review["decision"]]
            candidate["reuse_eligible"] = False
            if review["decision"] == "APPROVE":
                candidate["promotion_status"] = "APPROVED_PENDING_PROMOTION"
        return deepcopy(item)

    def mark_execution_experience_promoted(self, experience_id, memcube_id, promoted_at):
        item = self.candidates[experience_id]
        item["candidate"]["promotion_status"] = "ACTIVE"
        item["candidate"]["reuse_eligible"] = True
        item["candidate"]["memcube_id"] = memcube_id
        item["candidate"]["promoted_at"] = promoted_at
        return deepcopy(item)

    def deactivate_execution_experience_candidate(self, experience_id, **details):
        item = self.candidates[experience_id]
        item["candidate"]["reuse_eligible"] = False
        item["candidate"]["promotion_status"] = details["disposition"]
        item["candidate"]["deactivation_reason"] = details["reason"]
        return deepcopy(item)


class ExperienceMemCubeStoreFixture:
    def __init__(self):
        self.rows = {}

    def insert(self, data, *, if_exists="error"):
        row = deepcopy(data)
        if row["id"] not in self.rows:
            self.rows[row["id"]] = row
        elif if_exists == "error":
            raise ValueError("duplicate fixture MemCube")
        return SimpleNamespace(**deepcopy(self.rows[row["id"]]))

    def search(self, **filters):
        return [
            SimpleNamespace(**deepcopy(row))
            for row in self.rows.values()
            if all(row.get(key) == value for key, value in filters.items() if key in row)
        ]

    def set_refined_result(self, cube_id, *, extra_data=None, **_kwargs):
        row = self.rows.get(cube_id)
        if not row:
            return None
        row["extra_data"].update(extra_data or {})
        return SimpleNamespace(**deepcopy(row))


def accepted_experience_fixture():
    conversation_id = 404
    execution_id = "sex_" + "a" * 32
    worker_request_id = "wrq_" + "b" * 32
    trace_id = f"worker:{execution_id}"
    record = {
        "conversation_id": conversation_id,
        "execution_id": execution_id,
        "originating_message_id": 1001,
        "assistant_message_id": 1002,
        "worker_request_id": worker_request_id,
        "worker_work_id": "wk_fixture",
        "trace_id": trace_id,
        "status": "COMPLETED_UNVERIFIED",
        "backend": "opencode-docker",
        "requested_model": "opencode-go/gpt-6-luna",
        "provider_request_model": "gpt-6-luna",
        "provider_response_status": "200",
        "opencode_session_id": "ses_fixture",
        "container_id": "c" * 64,
        "image_digest": "sha256:" + "d" * 64,
        "project_id": "stn04-calculator",
        "workspace_id": "stn04-calculator",
        "objective": "Fix the arithmetic defect in calculator.py and run tests.",
        "verification_command": "python3 -m unittest",
        "verification_status": "passed",
        "verification_exit_code": "0",
        "artifacts": [{"id": "calculator.py", "kind": "modified_file"}],
        "result_summary": "Both arithmetic tests passed.",
        "acceptance_id": None,
    }
    trace = {
        "trace_id": trace_id,
        "execution_id": execution_id,
        "conversation_id": conversation_id,
        "user_message_id": 1001,
        "worker_request_id": worker_request_id,
        "backend": record["backend"],
        "requested_model": record["requested_model"],
    }
    messages = [SimpleNamespace(id=1001, role="user"), SimpleNamespace(id=1002, role="assistant")]
    memory = SimpleNamespace(
        chats=ExperienceChatFixture(record, messages),
        plan_traces=ExperienceTraceStoreFixture(trace),
        casebooks=ExperienceCaseBookStoreFixture(),
        memcubes=ExperienceMemCubeStoreFixture(),
        goals=SimpleNamespace(get_or_create=lambda _data: SimpleNamespace(id=701)),
    )
    service = ExecutionExperienceService(
        memory=memory,
        cfg={"experience_learning": {"enabled": True, "retrieval": {"enabled": True, "max_results": 3, "candidate_pool": 20, "max_context_chars": 1200}}},
    )
    return service, memory, conversation_id, execution_id


def test_worker_loopback_url_and_versioned_result_validation():
    with pytest.raises(WorkerExecutionError, match="loopback"):
        WorkerClient("http://169.254.169.254/latest/meta-data")
    with pytest.raises(WorkerExecutionError, match="loopback"):
        WorkerClient("https://127.0.0.1:7317")
    with pytest.raises(WorkerExecutionError, match="loopback"):
        WorkerClient("http://localhost:7317")
    result = {
        "schema": "worker.work-result.v1",
        "request_id": "wrq_0123456789abcdef0123456789abcdef",
        "status": "AWAITING_ACCEPTANCE",
        "execution": {"backend": "opencode"},
    }
    assert validate_work_result(result, expected_backend="opencode") == result
    with pytest.raises(WorkerExecutionError, match="did not match"):
        validate_work_result({**result, "status": "SUCCESS"}, expected_backend="opencode")
    with pytest.raises(WorkerExecutionError, match="did not match"):
        validate_work_result({**result, "request_id": "../other"}, expected_backend="opencode")
    with pytest.raises(WorkerExecutionError) as invalid_id:
        WorkerClient("http://127.0.0.1:7317", timeout_seconds=1).result("../other", backend="opencode")
    assert invalid_id.value.code == "WORKER_REQUEST_ID_INVALID"
    with patch.object(requests.Session, "request", side_effect=requests.Timeout("fixture timeout")):
        with pytest.raises(WorkerExecutionError) as error:
            WorkerClient("http://127.0.0.1:7317", timeout_seconds=1).health()
        assert error.value.code == "WORKER_TIMEOUT"

    unavailable = WorkerExecutionService(
        memory=SimpleNamespace(),
        cfg={"worker": {
            "enabled": True,
            "base_url": "http://127.0.0.1:1",
            "timeout_seconds": 1,
            "default_backend": "opencode",
            "allowed_backends": ["opencode"],
            "model": "fixture/provider-model",
            "readiness": {"model_verified": True},
            "workspaces": {"fixture": {"execution_boundary_verified": True}},
        }},
    ).readiness()
    assert unavailable["available"] is False
    assert unavailable["error_code"] == "WORKER_UNAVAILABLE"


def test_worker_client_retrieves_versioned_result_by_request_identity():
    result = {
        "schema": "worker.work-result.v1",
        "request_id": "wrq_0123456789abcdef0123456789abcdef",
        "status": "AWAITING_ACCEPTANCE",
        "execution": {"backend": "opencode"},
    }
    worker = FakeWorker()
    worker.jobs.append({"job_id": result["request_id"], "status": "AWAITING_ACCEPTANCE", "result": result})
    with RunningWorker(worker) as server:
        assert WorkerClient(server.url, timeout_seconds=1).result(
            result["request_id"], backend="opencode"
        ) == result


def test_worker_loopback_api_uses_no_bearer_authentication():
    worker = FakeWorker()
    with RunningWorker(worker) as server:
        client = WorkerClient(server.url, timeout_seconds=1)
        assert client.health()["security"]["api_auth_required"] is False
        assert client.jobs() == []
        result = client.run({
            "schema": "worker.work-request.v1",
            "objective": "loopback fixture without Worker bearer auth",
            "workspace": {"root": "C:/fixture", "project_id": "fixture-project"},
            "execution": {"backend": "opencode", "model": "fixture/provider-model"},
        })
        assert result["status"] == "AWAITING_ACCEPTANCE"
        assert all(value is None for value in worker.auth_headers)


def test_chat_template_compiles_with_explicit_worker_action_and_failure_warning():
    env = Environment(loader=FileSystemLoader("sis/templates"), autoescape=True)
    template = env.get_template("chats/detail.html")
    evidence_template = env.get_template("chats/execution_evidence.html")
    assert template.name == "chats/detail.html"
    assert evidence_template.name == "chats/execution_evidence.html"
    source = Path("sis/templates/chats/detail.html").read_text(encoding="utf-8")
    assert "Run with OpenCode" in source
    assert "does not guarantee cancellation" in source


def test_worker_result_context_keeps_bounded_external_provenance():
    history = [SimpleNamespace(
        role="assistant",
        text="Finding: the parser accepts an empty identifier.",
        meta={
            "runtime": "worker",
            "worker_execution_id": "sex_fixture",
            "worker_request_id": "wrq_0123456789abcdef0123456789abcdef",
            "requested_model": "fixture/provider-model",
            "status": "COMPLETED_UNVERIFIED",
        },
    )]
    messages = build_chat_messages(history, "What did Worker find?", {
        "system_prompt": "You are Stephanie.",
        "history_limit": 4,
        "max_context_chars": 1000,
        "max_context_message_chars": 180,
    })
    external = next(message for message in messages if message["role"] == "assistant")
    assert "External Worker result" in external["content"]
    assert "COMPLETED_UNVERIFIED" in external["content"]
    assert "not verified by Stephanie" in external["content"]
    assert len(external["content"]) <= 180


def test_authenticated_sis_submission_persists_chat_and_plantrace_link(tmp_path, monkeypatch):
    worker = FakeWorker()
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, sessions, engine = make_app(tmp_path, worker)
        monkeypatch.setattr(PlanTraceORM.__table__.c.domains, "type", JSON())
        trace_path = tmp_path / "worker-plan-traces.sqlite"
        trace_engine = create_engine(f"sqlite:///{trace_path}")
        create_sqlite_plan_trace_tables(trace_engine)
        trace_sessions = sessionmaker(bind=trace_engine, expire_on_commit=False)
        app.state.memory.plan_traces = PlanTraceStore(trace_sessions)
        configure_worker(config, server.url)
        response = client.post(
            f"/chats/{conversation_id}/executions",
            headers={"Idempotency-Key": "review-001"},
            json={
                "objective": "Inspect the fixture and report the bug.",
                "workspace_id": "fixture",
                    "backend": "opencode",
                    "permission_class": "inspect",
                    "dry_run": False,
            },
        )
        assert response.status_code == 202, response.text
        execution_id = response.json()["execution"]["execution_id"]
        record = wait_terminal(client, conversation_id, execution_id)

        evidence = client.get(f"/chats/{conversation_id}/executions/{execution_id}/evidence")
        assert evidence.status_code == 200
        assert "Worker execution evidence" in evidence.text
        assert execution_id in evidence.text
        detail = client.get(f"/chats/{conversation_id}")
        assert detail.status_code == 200
        assert "Run with OpenCode" in detail.text
        assert "does not guarantee cancellation" in detail.text
        assert "Disposable fixture" in detail.text

        assert record["status"] == "COMPLETED_UNVERIFIED"
        assert record["worker_request_id"] == "wrq_0123456789abcdef0123456789abcdef"
        assert record["opencode_session_id"] == "ses_fixture"
        assert record["requested_model"] == "fixture/provider-model"
        assert record["model"] is None  # Basic Worker result did not report an observed model identity.
        assert record["originating_message_id"]
        assert record["assistant_message_id"]
        assert record["originating_turn_id"]
        assert record["trace_id"] == f"worker:{execution_id}"
        assert worker.post_count == 1

        sent = worker.requests[0]
        assert sent["schema"] == "worker.work-request.v1"
        assert sent["execution"] == {"backend": "opencode", "model": "fixture/provider-model"}
        assert sent["workspace"]["root"] == str(tmp_path.resolve())
        assert sent["metadata"]["source"] == "sis"
        assert sent["metadata"]["operation_id"] == execution_id
        assert "messages" not in sent and "conversation" not in sent

        fresh_engine = create_engine(
            f"sqlite:///{tmp_path / 'sis-worker-test.sqlite'}",
            connect_args={"check_same_thread": False},
        )
        fresh_sessions = sessionmaker(bind=fresh_engine, expire_on_commit=False)
        reopened = ChatStore(fresh_sessions)
        persisted = reopened.get_external_execution(conversation_id, execution_id)
        assert persisted["worker_request_id"] == record["worker_request_id"]
        assert persisted["originating_message_id"] == record["originating_message_id"]
        conversation = reopened.get_conversation(conversation_id)
        assert execution_id in conversation.meta["sis_executions"]
        messages = reopened.get_messages(conversation_id)
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[0].meta["sis_execution_id"] == execution_id
        assert messages[1].meta["worker_request_id"] == record["worker_request_id"]
        assert len(reopened.get_turns_for_conversation(conversation_id)) == 1
        fresh_trace_engine = create_engine(f"sqlite:///{trace_path}")
        fresh_trace_sessions = sessionmaker(bind=fresh_trace_engine, expire_on_commit=False)
        trace = PlanTraceStore(fresh_trace_sessions).get_by_trace_id(record["trace_id"])
        assert trace is not None
        assert trace.meta["conversation_id"] == conversation_id
        assert [step.step_id for step in trace.execution_steps] == ["worker_submission", "worker_result"]
        assert trace.execution_steps[-1].step_type == "observation"
        assert trace.execution_steps[-1].meta["worker_request_id"] == record["worker_request_id"]
        app.state.worker_execution.shutdown()
        fresh_trace_engine.dispose()
        trace_engine.dispose()
        fresh_engine.dispose()
        engine.dispose()


def test_sis_worker_adapter_uses_loopback_worker_without_bearer_auth(tmp_path):
    worker = FakeWorker()
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, _sessions, engine = make_app(tmp_path, worker)
        configure_worker(config, server.url)
        response = client.post(
            f"/chats/{conversation_id}/executions",
            json={"objective": "authenticated controlled task", "workspace_id": "fixture", "backend": "opencode", "dry_run": False},
        )
        assert response.status_code == 202
        record = wait_terminal(client, conversation_id, response.json()["execution"]["execution_id"])
        assert record["status"] == "COMPLETED_UNVERIFIED"
        assert all(value is None for value in worker.auth_headers)
        app.state.worker_execution.shutdown()
        engine.dispose()


def test_authenticated_worker_dry_run_is_persisted_and_never_becomes_agent_success(tmp_path, monkeypatch):
    worker = FakeWorker()
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, sessions, engine = make_app(tmp_path, worker)
        monkeypatch.setattr(PlanTraceORM.__table__.c.domains, "type", JSON())
        trace_path = tmp_path / "worker-dry-run-traces.sqlite"
        trace_engine = create_engine(f"sqlite:///{trace_path}")
        create_sqlite_plan_trace_tables(trace_engine)
        trace_sessions = sessionmaker(bind=trace_engine, expire_on_commit=False)
        app.state.memory.plan_traces = PlanTraceStore(trace_sessions)
        configure_worker(config, server.url)
        config["workspaces"]["fixture"]["execution_boundary_verified"] = False
        config["workspaces"]["fixture"]["dry_run_only"] = True
        response = client.post(
            f"/chats/{conversation_id}/executions",
            headers={"Idempotency-Key": "dry-run-001"},
            json={
                "objective": "Verify authenticated Worker transport without invoking OpenCode.",
                "workspace_id": "fixture",
                "backend": "opencode",
                "permission_class": "inspect",
                "dry_run": True,
            },
        )
        assert response.status_code == 202, response.text
        execution_id = response.json()["execution"]["execution_id"]
        record = wait_terminal(client, conversation_id, execution_id)
        assert record["status"] == "DRY_RUN_NO_AGENT_EXECUTION"
        assert record["status_label"] == "DRY RUN — NO AGENT EXECUTION"
        assert record["dry_run"] is True
        assert record["worker_request_id"]
        assert record["opencode_session_id"] is None
        assert record.get("assistant_message_id") is None
        assert len(ChatStore(sessions).get_turns_for_conversation(conversation_id)) == 0
        assert worker.post_count == 1
        assert worker.health_calls == 0
        assert worker.model_discovery_calls == 0
        assert worker.transport_health_calls >= 1
        fresh_chat_engine = create_engine(
            f"sqlite:///{tmp_path / 'sis-worker-test.sqlite'}",
            connect_args={"check_same_thread": False},
        )
        fresh_chat_sessions = sessionmaker(bind=fresh_chat_engine, expire_on_commit=False)
        fresh_chat = ChatStore(fresh_chat_sessions)
        persisted = fresh_chat.get_external_execution(conversation_id, execution_id)
        assert persisted["status"] == "DRY_RUN_NO_AGENT_EXECUTION"
        assert persisted["worker_request_id"] == record["worker_request_id"]
        assert fresh_chat.get_messages(conversation_id)[0].meta["sis_execution_dry_run"] is True
        assert fresh_chat.get_turns_for_conversation(conversation_id) == []
        sent = worker.requests[0]
        assert sent["options"]["dry_run"] is True
        assert sent["execution"] == {"backend": "opencode", "model": "fixture/provider-model"}
        assert worker.health_calls == 0
        assert worker.model_discovery_calls == 0
        assert worker.transport_health_calls >= 1
        fresh_trace_engine = create_engine(f"sqlite:///{trace_path}")
        fresh_trace_sessions = sessionmaker(bind=fresh_trace_engine, expire_on_commit=False)
        trace = PlanTraceStore(fresh_trace_sessions).get_by_trace_id(record["trace_id"])
        assert trace is not None
        assert trace.meta["status"] == "DRY_RUN_NO_AGENT_EXECUTION"
        assert trace.meta["dry_run"] is True
        assert trace.execution_steps[-1].step_id == "worker_result"
        evidence = client.get(f"/chats/{conversation_id}/executions/{execution_id}/evidence")
        assert evidence.status_code == 200
        assert "DRY RUN — NO AGENT EXECUTION" in evidence.text
        app.state.worker_execution.shutdown()
        fresh_chat_engine.dispose()
        fresh_trace_engine.dispose()
        trace_engine.dispose()
        engine.dispose()


def test_dry_run_only_route_works_with_chat_and_general_execution_disabled(tmp_path):
    worker = FakeWorker()
    with RunningWorker(worker) as server:
        app, client, _existing_conversation_id, config, _traces, sessions, engine = make_app(
            tmp_path, worker, enabled=False, chat_enabled=False
        )
        configure_worker(config, server.url)
        config["dry_run_only"] = True
        config["workspaces"]["fixture"]["execution_boundary_verified"] = False
        config["workspaces"]["fixture"]["dry_run_only"] = True
        response = client.post(
            "/chats/worker-dry-runs",
            headers={"Idempotency-Key": "ops-dry-run-001"},
            json={
                "title": "STN03 real-route dry-run fixture",
                "objective": "Verify Worker transport without invoking any agent.",
                "workspace_id": "fixture",
                "backend": "opencode",
            },
        )
        assert response.status_code == 202, response.text
        payload = response.json()
        conversation_id = payload["conversation_id"]
        execution_id = payload["execution"]["execution_id"]
        record = wait_terminal(client, conversation_id, execution_id)
        assert record["status"] == "DRY_RUN_NO_AGENT_EXECUTION"
        assert record["dry_run"] is True
        assert payload["label"] == "DRY RUN — NO AGENT EXECUTION"
        assert worker.post_count == 1
        assert worker.health_calls == 0
        assert worker.model_discovery_calls == 0
        assert worker.transport_health_calls >= 1
        assert worker.requests[0]["options"]["dry_run"] is True
        conversation = ChatStore(sessions).get_conversation(conversation_id)
        assert conversation.meta["sis_owner"] == "local"
        assert conversation.meta["sis_dry_run_only"] is True
        assert ChatStore(sessions).get_turns_for_conversation(conversation_id) == []
        normal_execution = client.post(
            f"/chats/{conversation_id}/executions",
            headers={"Idempotency-Key": "ops-not-real-001"},
            json={
                "objective": "must not execute",
                "workspace_id": "fixture",
                "backend": "opencode",
                "dry_run": False,
            },
        )
        assert normal_execution.status_code == 404
        rendered_context = {}
        original_template_response = app.state.templates.TemplateResponse
        def capture_template_context(name, context, **kwargs):
            rendered_context.update(context)
            return original_template_response(name, context, **kwargs)
        app.state.templates.TemplateResponse = capture_template_context
        detail = client.get(payload["conversation_url"])
        assert detail.status_code == 200
        assert rendered_context["worker_status"]["configured"] is True
        assert "Worker transport dry-run" in detail.text
        assert record["execution_id"] in detail.text
        assert "DRY RUN — NO AGENT EXECUTION" in detail.text
        app.state.worker_execution.shutdown()
        engine.dispose()


def test_worker_submission_uses_sis_auth_and_workspace_without_chat_model_gate(tmp_path):
    worker = FakeWorker()
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, _sessions, engine = make_app(tmp_path, worker)
        configure_worker(config, server.url)
        path = f"/chats/{conversation_id}/executions"
        body = {"objective": "inspect", "workspace_id": "fixture", "backend": "opencode"}

        no_csrf = client.post(path, json=body, headers={"X-CSRF-Token": "bad"})
        assert no_csrf.status_code == 403
        assert worker.post_count == 0

        app.state.cfg["chat_runtime"]["enabled"] = False
        created = client.post(
            "/chats",
            headers={"Idempotency-Key": "worker-only-conversation"},
            json={"title": "Docker task fixture"},
        )
        assert created.status_code == 201, created.text
        worker_only_id = created.json()["conversation"]["id"]
        worker_only_submission = client.post(
            f"/chats/{worker_only_id}/executions",
            headers={"Idempotency-Key": "worker-only-task"},
            json={**body, "dry_run": False},
        )
        assert worker_only_submission.status_code == 202
        assert worker_only_submission.json()["execution"]["conversation_id"] == worker_only_id
        worker_only_record = wait_terminal(
            client, worker_only_id, worker_only_submission.json()["execution"]["execution_id"]
        )
        assert worker_only_record["status"] == "COMPLETED_UNVERIFIED"
        assert worker.post_count == 1
        app.state.cfg["chat_runtime"]["enabled"] = True

        not_registered = client.post(path, json={**body, "workspace_id": "other"})
        assert not_registered.status_code == 403
        assert worker.post_count == 1

        owner_id = app.state.memory.chats.add_conversation({
            "provider": "stephanie", "title": "not owned", "meta": {"sis_owner": "someone-else"},
        }).id
        unauthorized = client.post(f"/chats/{owner_id}/executions", json=body)
        assert unauthorized.status_code == 404
        assert worker.post_count == 1
        app.state.worker_execution.shutdown()
        engine.dispose()


def test_feature_disabled_executor_unavailable_and_explicit_backend_fail_closed(tmp_path):
    worker = FakeWorker(available=False)
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, _sessions, engine = make_app(tmp_path, worker, enabled=False)
        configure_worker(config, server.url)
        disabled = client.post(
            f"/chats/{conversation_id}/executions",
            json={"objective": "inspect", "workspace_id": "fixture", "backend": "opencode", "dry_run": False},
        )
        assert disabled.status_code == 404
        assert worker.post_count == 0

        app.state.cfg["worker"]["enabled"] = True
        readiness = client.get(f"/chats/{conversation_id}/executions/readiness")
        assert readiness.status_code == 200
        assert readiness.json()["available"] is False
        unavailable = client.post(
            f"/chats/{conversation_id}/executions",
            json={"objective": "inspect", "workspace_id": "fixture", "backend": "opencode", "dry_run": False},
        )
        assert unavailable.status_code == 503
        worker.available = True
        app.state.cfg["worker"]["readiness"]["model_verified"] = False
        unverified_model = app.state.worker_execution.readiness(dry_run=False)
        assert unverified_model["model_discovered"] is True
        assert unverified_model["model_verified"] is False
        blocked_model = client.post(
            f"/chats/{conversation_id}/executions",
            json={"objective": "inspect", "workspace_id": "fixture", "backend": "opencode", "dry_run": False},
        )
        assert blocked_model.status_code == 503
        app.state.cfg["worker"]["readiness"]["model_verified"] = True
        disallowed = client.post(
            f"/chats/{conversation_id}/executions",
            json={"objective": "inspect", "workspace_id": "fixture", "backend": "other"},
        )
        assert disallowed.status_code == 422
        assert worker.post_count == 0
        app.state.worker_execution.shutdown()
        engine.dispose()


def test_execution_experience_requires_human_acceptance_then_keeps_candidate_pending_until_second_review():
    service, memory, conversation_id, execution_id = accepted_experience_fixture()
    with pytest.raises(ExperienceError) as pending:
        service.extract_candidate(conversation_id=conversation_id, execution_id=execution_id)
    assert pending.value.code == "HUMAN_ACCEPTANCE_REQUIRED"
    assert memory.casebooks.candidates == {}
    assert memory.memcubes.rows == {}

    evidence = service.review_evidence(conversation_id, execution_id)
    refs = evidence["evidence_refs"]
    accepted = service.review_execution(
        conversation_id=conversation_id,
        execution_id=execution_id,
        reviewer="local-reviewer",
        decision="ACCEPT",
        review_id="review-accept-001",
        notes="Reviewed the narrow arithmetic correction and independent test result.",
        evidence_refs=refs,
        accepted_scope=["calculator.py"],
    )
    assert accepted["human_review"]["decision"] == "ACCEPT"
    assert accepted["acceptance_id"] is None
    assert accepted["execution_status"] == "COMPLETED_UNVERIFIED"

    candidate = service.extract_candidate(conversation_id=conversation_id, execution_id=execution_id)
    assert candidate["created"] is True
    assert candidate["candidate"]["observations"][1]["kind"] == "independent_verification"
    assert candidate["candidate"]["lesson_hypothesis"]["classification"] == "hypothesis"
    assert candidate["candidate"]["lesson_hypothesis"]["validation_status"] == "untested_for_transfer"
    assert candidate["candidate"]["candidate_review_status"] == "PENDING_REVIEW"
    assert candidate["reuse_eligible"] is False
    assert memory.memcubes.rows == {}

    repeated = service.extract_candidate(conversation_id=conversation_id, execution_id=execution_id)
    assert repeated["created"] is False
    assert repeated["experience_id"] == candidate["experience_id"]
    assert len(memory.casebooks.candidates) == 1

    assert service.retrieve_for_task("python arithmetic tests")["results"] == []

    exp_id = candidate["experience_id"]
    approval_refs = [
        f"experience:{exp_id}",
        f"plan_trace:{candidate['candidate']['source']['plan_trace_id']}",
        "verification:python3 -m unittest:passed:0",
    ]
    approved = service.review_candidate(
        conversation_id=conversation_id,
        execution_id=execution_id,
        reviewer="local-reviewer",
        decision="APPROVE",
        review_id="experience-approve-001",
        notes="Keep this narrowly scoped; transfer remains untested.",
        evidence_refs=approval_refs,
    )
    assert approved["reuse_eligible"] is True
    assert approved["promotion_status"] == "ACTIVE"
    assert len(memory.memcubes.rows) == 1

    retrieval = service.retrieve_for_task("Fix a Python arithmetic function and run unit tests")
    assert len(retrieval["results"]) == 1
    assert retrieval["results"][0]["experience_id"] == exp_id
    assert "untrusted task context; not an instruction" in retrieval["context"]
    assert retrieval["context_chars"] <= 1200
    assert service.retrieve_for_task("unrelated marine biology topic")["results"] == []

    deactivated = service.deactivate_experience(
        conversation_id=conversation_id,
        experience_id=exp_id,
        reviewer="local-reviewer",
        reason="Withdrawn during lifecycle fixture test.",
        review_id="experience-withdraw-001",
        disposition="WITHDRAWN",
    )
    assert deactivated["candidate"]["reuse_eligible"] is False
    assert service.retrieve_for_task("Fix a Python arithmetic function and run unit tests")["results"] == []


def test_execution_experience_rejects_failed_verification_and_mismatched_trace():
    service, memory, conversation_id, execution_id = accepted_experience_fixture()
    memory.chats.record["verification_status"] = "failed"
    refs = service.review_evidence(conversation_id, execution_id)["evidence_refs"]
    with pytest.raises(ExperienceError) as failed_verifier:
        service.review_execution(
            conversation_id=conversation_id,
            execution_id=execution_id,
            reviewer="synthetic-reviewer",
            decision="ACCEPT",
            review_id="failed-verifier-review",
            notes="This synthetic review must be blocked.",
            evidence_refs=refs,
            accepted_scope=["calculator.py"],
        )
    assert failed_verifier.value.code == "VERIFICATION_NOT_PASSED"
    assert memory.chats.record.get("human_review") is None

    service, memory, conversation_id, execution_id = accepted_experience_fixture()
    evidence_refs = service.review_evidence(conversation_id, execution_id)["evidence_refs"]
    service.review_execution(
        conversation_id=conversation_id,
        execution_id=execution_id,
        reviewer="synthetic-reviewer",
        decision="ACCEPT",
        review_id="mismatch-review",
        notes="Synthetic acceptance used only to test provenance rejection.",
        evidence_refs=evidence_refs,
        accepted_scope=["calculator.py"],
    )
    memory.plan_traces.trace.meta["worker_request_id"] = "wrq_wrong"
    with pytest.raises(ExperienceError) as mismatch:
        service.extract_candidate(conversation_id=conversation_id, execution_id=execution_id)
    assert mismatch.value.code == "PROVENANCE_MISMATCH"
    assert memory.casebooks.candidates == {}


def test_rejected_experience_candidate_is_never_promoted():
    service, memory, conversation_id, execution_id = accepted_experience_fixture()
    evidence_refs = service.review_evidence(conversation_id, execution_id)["evidence_refs"]
    service.review_execution(
        conversation_id=conversation_id,
        execution_id=execution_id,
        reviewer="synthetic-reviewer",
        decision="ACCEPT",
        review_id="execution-accept-for-candidate-reject-test",
        notes="Synthetic acceptance for a candidate rejection test.",
        evidence_refs=evidence_refs,
        accepted_scope=["calculator.py"],
    )
    candidate = service.extract_candidate(conversation_id=conversation_id, execution_id=execution_id)
    experience_id = candidate["experience_id"]
    rejected = service.review_candidate(
        conversation_id=conversation_id,
        execution_id=execution_id,
        reviewer="synthetic-reviewer",
        decision="REJECT",
        review_id="candidate-reject-test",
        notes="Synthetic rejection; never promoted.",
        evidence_refs=[f"experience:{experience_id}"],
    )
    assert rejected["candidate_review_status"] == "REJECTED"
    assert rejected["reuse_eligible"] is False
    assert memory.memcubes.rows == {}
    assert service.retrieve_for_task("Python arithmetic unit tests")["results"] == []


def test_execution_review_is_authenticated_csrf_protected_and_does_not_create_worker_acceptance(tmp_path):
    worker = FakeWorker()
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, _sessions, engine = make_app(tmp_path, worker)
        configure_worker(config, server.url)
        execution_id = "sex_" + "a" * 32
        worker_request_id = "wrq_" + "b" * 32
        trace_id = f"worker:{execution_id}"
        record = {
            "execution_id": execution_id,
            "conversation_id": conversation_id,
            "originating_message_id": None,
            "username": "local",
            "project_id": "fixture-project",
            "workspace_id": "fixture",
            "backend": "opencode-docker",
            "requested_model": "opencode-go/gpt-6-luna",
            "objective": "Fix the fixture.",
            "status": "COMPLETED_UNVERIFIED",
            "trace_id": trace_id,
            "worker_request_id": worker_request_id,
            "verification_command": "python3 -m unittest",
            "verification_status": "passed",
            "verification_exit_code": "0",
            "artifacts": [{"id": "calculator.py", "kind": "modified_file"}],
            "acceptance_id": None,
        }
        message, stored, created = app.state.memory.chats.start_external_execution(
            conversation_id, "review-fixture-key", "fingerprint", "Fix the fixture.", record
        )
        assert created is True
        app.state.memory.chats.update_external_execution(
            conversation_id, execution_id, {"originating_message_id": message.id}
        )
        path = f"/chats/{conversation_id}/executions/{execution_id}/review"
        no_csrf = client.post(path, json={"decision": "ACCEPT", "accepted_scope": ["calculator.py"]}, headers={"X-CSRF-Token": "bad"})
        assert no_csrf.status_code == 403
        refs = [
            f"worker_request:{worker_request_id}",
            f"plan_trace:{trace_id}",
            "verification:python3 -m unittest:passed:0",
            "artifact:calculator.py",
        ]
        response = client.post(
            path,
            headers={"Idempotency-Key": "review-human-accept-001"},
            json={"decision": "ACCEPT", "notes": "Explicit test decision.", "evidence_refs": refs, "accepted_scope": ["calculator.py"]},
        )
        assert response.status_code == 200, response.text
        review = response.json()["review"]
        assert review["human_review"]["decision"] == "ACCEPT"
        assert review.get("human_review_history"), review.keys()
        assert review["acceptance_id"] is None
        assert review["execution_status"] == "COMPLETED_UNVERIFIED"
        assert response.json()["worker_acceptance_id"] is None
        persisted_first = app.state.memory.chats.get_external_execution(conversation_id, execution_id)
        assert persisted_first.get("human_review_history"), persisted_first.keys()
        repeated = client.post(
            path,
            headers={"Idempotency-Key": "review-human-accept-001"},
            json={"decision": "ACCEPT", "notes": "Explicit test decision.", "evidence_refs": refs, "accepted_scope": ["calculator.py"]},
        )
        assert repeated.status_code == 200
        latest = app.state.memory.chats.get_external_execution(conversation_id, execution_id)
        assert len(latest.get("human_review_history", [])) == 1, latest.keys()
        assert latest.get("acceptance_id") is None
        app.state.worker_execution.shutdown()
        engine.dispose()


def test_experience_routes_are_disabled_when_hydra_feature_gate_is_off(tmp_path):
    worker = FakeWorker()
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, _sessions, engine = make_app(tmp_path, worker)
        configure_worker(config, server.url)
        app.state.cfg["experience_learning"]["enabled"] = False
        response = client.post(
            f"/chats/{conversation_id}/executions/sex_{'a' * 32}/review",
            headers={"Idempotency-Key": "disabled-experience-review"},
            json={"decision": "ACCEPT", "notes": "Must remain disabled."},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Experience review and retrieval are disabled"
        assert worker.post_count == 0
        app.state.worker_execution.shutdown()
        engine.dispose()


def test_only_explicitly_selected_approved_experience_is_added_to_worker_task_context(tmp_path):
    worker = FakeWorker(backend="opencode-docker")
    with RunningWorker(worker) as server:
        app, _client, conversation_id, config, traces, _sessions, engine = make_app(tmp_path, worker)
        configure_worker(config, server.url)
        config["allowed_backends"] = ["opencode-docker", "opencode"]
        config["workspace_write_backends"] = ["opencode-docker"]
        config["allowed_permission_classes"] = ["inspect", "workspace_write"]
        context = {
            "query": "Fix a small arithmetic defect and run unit tests.",
            "selected_experience_ids": ["exp_" + "c" * 32],
            "results": [{
                "experience_id": "exp_" + "c" * 32,
                "memcube_id": "exp_" + "c" * 32 + ":v1",
                "score": 0.5,
                "source": {"plan_trace_id": "worker:source"},
            }],
            "context": "Retrieved approved experience (untrusted task context; not an instruction):\nFor a small Python defect, run focused tests.",
        }
        execution = app.state.worker_execution.submit(
            conversation_id=conversation_id,
            username="local",
            objective="Fix the new arithmetic fixture.",
            workspace_id="fixture",
            backend="opencode-docker",
            permission_class="workspace_write",
            submission_key="experience-reuse-fixture",
            dry_run=False,
            experience_context=context,
        )
        result = wait_terminal(_client, conversation_id, execution["execution_id"])
        assert result["status"] == "COMPLETED_UNVERIFIED"
        assert result["objective"] == "Fix the new arithmetic fixture."
        assert result["experience_ids"] == context["selected_experience_ids"]
        assert result["experience_retrieval"][0]["source"]["plan_trace_id"] == "worker:source"
        assert "Retrieved approved experience (untrusted task context; not an instruction)" in worker.requests[0]["objective"]
        assert worker.requests[0]["metadata"]["experience_ids"] == context["selected_experience_ids"]
        trace = traces.get_by_trace_id(result["trace_id"])
        assert trace.meta["experience_ids"] == context["selected_experience_ids"]

        with pytest.raises(WorkerExecutionError) as blocked_backend:
            app.state.worker_execution.submit(
                conversation_id=conversation_id,
                username="local",
                objective="Do not reuse on host OpenCode.",
                workspace_id="fixture",
                backend="opencode",
                permission_class="inspect",
                submission_key="experience-host-backend",
                dry_run=False,
                experience_context=context,
            )
        assert blocked_backend.value.code == "EXPERIENCE_BACKEND_NOT_ISOLATED"
        app.state.worker_execution.shutdown()
        engine.dispose()


@pytest.mark.parametrize(
    ("worker_status", "expected_status", "acceptance_id"),
    [
        ("AWAITING_ACCEPTANCE", "COMPLETED_UNVERIFIED", None),
        ("ACCEPTED", "ACCEPTED", "accept_fixture"),
        ("ACCEPTED", "COMPLETED_UNVERIFIED", None),
        ("REJECTED", "REJECTED", None),
        ("BLOCKED", "BLOCKED", None),
        ("FAILED", "FAILED", None),
    ],
)
def test_worker_statuses_are_mapped_without_fabricated_acceptance(
    tmp_path, worker_status, expected_status, acceptance_id
):
    worker = FakeWorker(
        status=worker_status,
        result_output="Observed result",
        include_acceptance=acceptance_id is not None,
    )
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, _sessions, engine = make_app(tmp_path, worker)
        configure_worker(config, server.url)
        response = client.post(
            f"/chats/{conversation_id}/executions",
            headers={"Idempotency-Key": "map-status"},
            json={"objective": "inspect", "workspace_id": "fixture", "backend": "opencode", "dry_run": False},
        )
        assert response.status_code == 202
        record = wait_terminal(client, conversation_id, response.json()["execution"]["execution_id"])
        assert record["status"] == expected_status
        assert (record.get("acceptance_id") or None) == acceptance_id, {"record": record, "worker_result": worker.jobs[0].get("result") if worker.jobs else None}
        app.state.worker_execution.shutdown()
        engine.dispose()


def test_duplicate_submission_is_idempotent_and_worker_lost_response_is_not_retried(tmp_path):
    worker = FakeWorker()
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, _sessions, engine = make_app(tmp_path, worker)
        configure_worker(config, server.url)
        path = f"/chats/{conversation_id}/executions"
        headers = {"Idempotency-Key": "same-request"}
        body = {"objective": "inspect once", "workspace_id": "fixture", "backend": "opencode", "dry_run": False}
        first = client.post(path, headers=headers, json=body)
        replay = client.post(path, headers=headers, json=body)
        assert first.status_code == replay.status_code == 202
        assert first.json()["execution"]["execution_id"] == replay.json()["execution"]["execution_id"]
        wait_terminal(client, conversation_id, first.json()["execution"]["execution_id"])
        assert worker.post_count == 1
        conflict = client.post(path, headers=headers, json={**body, "objective": "different task"})
        assert conflict.status_code == 409
        app.state.worker_execution.shutdown()
        engine.dispose()

    lost = FakeWorker(post_status=503)
    with RunningWorker(lost) as server:
        app, client, conversation_id, config, _traces, _sessions, engine = make_app(tmp_path / "lost", lost)
        configure_worker(config, server.url)
        response = client.post(
            f"/chats/{conversation_id}/executions",
            json={"objective": "inspect once", "workspace_id": "fixture", "backend": "opencode", "dry_run": False},
        )
        assert response.status_code == 202
        record = wait_terminal(client, conversation_id, response.json()["execution"]["execution_id"])
        assert record["status"] == "INTERRUPTED"
        assert record["error_category"] == "OUTCOME_UNKNOWN"
        assert lost.post_count == 1
        app.state.worker_execution.shutdown()
        engine.dispose()


def test_parallel_duplicate_and_late_worker_result_reconcile_without_redispatch(tmp_path):
    worker = FakeWorker(post_status=503, retain_lost_result=True)
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, _sessions, engine = make_app(tmp_path, worker)
        configure_worker(config, server.url)
        path = f"/chats/{conversation_id}/executions"
        body = {"objective": "parallel duplicate", "workspace_id": "fixture", "backend": "opencode", "dry_run": False}

        def submit():
            return client.post(path, headers={"Idempotency-Key": "parallel-key"}, json=body)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = [future.result() for future in [pool.submit(submit), pool.submit(submit)]]
        assert first.status_code == second.status_code == 202
        execution_id = first.json()["execution"]["execution_id"]
        assert second.json()["execution"]["execution_id"] == execution_id
        record = wait_terminal(client, conversation_id, execution_id)
        assert record["status"] == "COMPLETED_UNVERIFIED"
        assert record["worker_request_id"] == "wrq_0123456789abcdef0123456789abcdef"
        independent = client.post(
            path,
            headers={"Idempotency-Key": "independent-request"},
            json={"objective": "independent task", "workspace_id": "fixture", "backend": "opencode", "dry_run": False},
        )
        assert independent.status_code == 202
        independent_record = wait_terminal(client, conversation_id, independent.json()["execution"]["execution_id"])
        assert independent_record["execution_id"] != execution_id
        assert independent_record["worker_request_id"] != record["worker_request_id"]
        assert worker.post_count == 2
        app.state.worker_execution.shutdown()
        engine.dispose()


def test_credential_like_objective_is_rejected_and_result_secrets_are_redacted(tmp_path):
    worker = FakeWorker(result_output="api_key=fixture-secret-value")
    with RunningWorker(worker) as server:
        app, client, conversation_id, config, _traces, sessions, engine = make_app(tmp_path, worker)
        configure_worker(config, server.url)
        path = f"/chats/{conversation_id}/executions"
        rejected = client.post(
            path,
            json={"objective": "inspect api_key=do-not-send-this", "workspace_id": "fixture", "backend": "opencode"},
        )
        assert rejected.status_code == 422
        assert worker.post_count == 0
        response = client.post(
            path,
            json={"objective": "inspect the fixture", "workspace_id": "fixture", "backend": "opencode", "dry_run": False},
        )
        record = wait_terminal(client, conversation_id, response.json()["execution"]["execution_id"])
        assert record.get("result_summary") is not None, {
            "status": record.get("status"),
            "dry_run": record.get("dry_run"),
            "error_category": record.get("error_category"),
            "worker_message": record.get("worker_message"),
            "worker_request": worker.requests[0] if worker.requests else None,
            "worker_result": worker.jobs[0].get("result") if worker.jobs else None,
        }
        assert "fixture-secret-value" not in record["result_summary"]
        persisted = ChatStore(sessions).get_external_execution(conversation_id, record["execution_id"])
        assert "fixture-secret-value" not in persisted["result_summary"]
        app.state.worker_execution.shutdown()
        engine.dispose()
