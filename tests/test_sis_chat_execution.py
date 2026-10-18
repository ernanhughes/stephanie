from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from jinja2 import Environment, FileSystemLoader
import pytest
import httpx
from omegaconf import OmegaConf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sis.routes.chats import _build_chat_messages, router
from sis.services.chat_execution import (
    ChatInferenceFailed,
    ConversationExecutionService,
    IdempotencyConflict,
)
from sis.template_utils import SISTemplates
from stephanie.memory.chat_store import ChatStore
from stephanie.memory.memory_tool import MemoryTool
from stephanie.services.registry_loader import load_services_profile
from stephanie.services.plan_trace_service import PlanTraceService
from stephanie.services.llm_service import LLMService
from stephanie.services.service_container import ServiceContainer
from stephanie.supervisor import Supervisor
from stephanie.utils.secret_utils import redact_secrets
from stephanie.orm.chat import (
    ChatConversationORM,
    ChatMessageORM,
    ChatTurnORM,
)
from stephanie.orm.base import (
    create_database_engine,
    create_database_engine_from_hydra,
    configure_legacy_orm,
    database_connection_parameters,
    database_url_from_config,
    dispose_legacy_orm,
    SessionLocal,
)


@dataclass
class FakeMessage:
    id: int
    conversation_id: int
    role: str
    text: str
    order_index: int
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "text": self.text,
            "order_index": self.order_index,
            "meta": self.meta,
        }


class FakeChatStore:
    def __init__(self):
        self.conversation = SimpleNamespace(
            id=17,
            meta={},
            to_dict=lambda: {"id": 17, "title": "Existing conversation"},
        )
        self.messages = [
            FakeMessage(1, 17, "user", "Earlier question", 0),
            FakeMessage(2, 17, "assistant", "Earlier answer", 1),
            FakeMessage(3, 17, "tool", "Untrusted tool payload", 2),
        ]
        self.turns = []
        self.next_id = 4
        self.created_conversation = None

    def add_conversation(self, data):
        self.created_conversation = data
        return SimpleNamespace(
            id=18,
            to_dict=lambda: {"id": 18, "title": data["title"]},
        )

    def add_conversation_once(self, data, request_id):
        existing = getattr(self, "created_request_id", None)
        if existing == request_id:
            if self.created_conversation["title"] != data["title"]:
                raise ValueError("Idempotency key was already used for another conversation")
        else:
            self.created_request_id = request_id
            self.created_conversation = {
                **data,
                "meta": {**(data.get("meta") or {}), "sis_create_request_id": request_id},
            }
        return SimpleNamespace(
            id=18,
            to_dict=lambda: {"id": 18, "title": data["title"]},
        )

    def get_conversation(self, conversation_id):
        return self.conversation if conversation_id == 17 else None

    def get_messages(self, conversation_id):
        assert conversation_id == 17
        return list(self.messages)

    def add_messages(self, conversation_id, messages):
        assert conversation_id == 17
        created = []
        for message in messages:
            item = FakeMessage(
                id=self.next_id,
                conversation_id=conversation_id,
                role=message["role"],
                text=message.get("text", ""),
                order_index=len(self.messages),
                meta=message.get("meta", {}),
            )
            self.next_id += 1
            self.messages.append(item)
            created.append(item)
        return created

    def add_turns(self, conversation_id, messages):
        self.turns.append((conversation_id, messages))

    def get_message_by_request_id(self, conversation_id, request_id, *, role=None):
        return next((
            message for message in self.messages
            if message.conversation_id == conversation_id
            and (message.meta or {}).get("sis_request_id") == request_id
            and (role is None or message.role == role)
        ), None)

    def get_messages_before(self, conversation_id, before_order_index, *, limit):
        return [
            message for message in self.messages
            if message.conversation_id == conversation_id
            and message.order_index < before_order_index
            and message.role in {"user", "assistant"}
        ][-limit:] if limit else []

    def add_user_message_once(self, conversation_id, request_id, text, *, meta=None):
        existing = self.get_message_by_request_id(conversation_id, request_id, role="user")
        if existing:
            if existing.text != text:
                raise ValueError("Idempotency key was already used for another message")
            return existing
        message = FakeMessage(
            self.next_id, conversation_id, "user", text, len(self.messages),
            {**(meta or {}), "sis_request_id": request_id, "execution_status": "in_progress"},
        )
        self.next_id += 1
        self.messages.append(message)
        return message

    def mark_user_message_status(self, conversation_id, message_id, status, *, error_type=None):
        message = next(item for item in self.messages if item.id == message_id)
        message.meta = {**message.meta, "execution_status": status}
        if error_type:
            message.meta["error_type"] = error_type
        return message

    def complete_assistant_turn(self, conversation_id, user_message_id, request_id, text, *, meta=None):
        existing = self.get_message_by_request_id(conversation_id, request_id, role="assistant")
        if existing:
            turn = next(turn for turn in self.turns if turn.assistant_message_id == existing.id)
            return existing, turn
        assistant = FakeMessage(
            self.next_id, conversation_id, "assistant", text, len(self.messages),
            {**(meta or {}), "sis_request_id": request_id, "execution_status": "completed"},
        )
        self.next_id += 1
        self.messages.append(assistant)
        user = next(item for item in self.messages if item.id == user_message_id)
        user.meta = {**user.meta, "execution_status": "completed"}
        turn = SimpleNamespace(
            id=len(self.turns) + 1,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            order_index=assistant.order_index,
            to_dict=lambda include_messages=False: {
                "id": len(self.turns),
                "user_message_id": user.id,
                "assistant_message_id": assistant.id,
                "order_index": assistant.order_index,
            },
        )
        self.turns.append(turn)
        return assistant, turn

    def get_turns_for_conversation(self, conversation_id):
        return list(self.turns)


class FakeLLM:
    def __init__(self, *, failure=False):
        self.failure = failure
        self.messages = None
        self.agent_cfg = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        self.agent_cfg = kwargs["agent_cfg"]
        if self.failure:
            raise RuntimeError("provider error containing private details")
        return {"text": "A verified test response", "model_name": "test/model", "cached": False}


class FakeContainer:
    def __init__(self, llm):
        self.llm = llm

    def get(self, name):
        if name != "llm":
            raise KeyError(name)
        return self.llm


class FakeLogger:
    def log(self, *_args, **_kwargs):
        pass


class LifecycleService:
    def __init__(self, *, is_async=False):
        self.is_async = is_async
        self.initialized = False
        self.stopped = False

    @property
    def name(self):
        return "lifecycle-test"

    def initialize(self):
        if self.is_async:
            async def initialize_async():
                self.initialized = True
            return initialize_async()
        self.initialized = True

    def health_check(self):
        return {"status": "healthy" if self.initialized else "unhealthy"}

    def shutdown(self):
        if self.is_async:
            async def shutdown_async():
                self.stopped = True
            return shutdown_async()
        self.stopped = True


def make_app(*, enabled=True, failure=False, llm=None):
    app = FastAPI()
    app.include_router(router)
    app.state.cfg = {
        "chat_runtime": {
            "enabled": enabled,
            "history_limit": 2,
            "max_context_chars": 1000,
            "max_context_message_chars": 100,
            "system_prompt": "Use evidence; be clear about uncertainty.",
            "model": {"name": "test/model", "api_base": "http://localhost"},
            "readiness": {"model_verified": True, "live_persistence_verified": True},
        }
    }
    app.state.memory = SimpleNamespace(chats=FakeChatStore())
    app.state.container = FakeContainer(llm or FakeLLM(failure=failure))
    app.state.chat_execution = ConversationExecutionService(
        memory=app.state.memory,
        container=app.state.container,
        cfg=app.state.cfg,
        logger=FakeLogger(),
    )
    app.state.auth_settings = SimpleNamespace(ready=True)
    app.state.database_ready = True
    @app.middleware("http")
    async def test_identity(request, call_next):
        request.state.username = "local"
        return await call_next(request)
    return app


def test_chat_context_is_bounded_and_excludes_tool_messages():
    history = [
        FakeMessage(1, 17, "user", "older", 0),
        FakeMessage(2, 17, "assistant", "recent", 1),
        FakeMessage(3, 17, "tool", "ignore this", 2),
    ]

    messages = _build_chat_messages(
        history,
        "latest",
        {
            "system_prompt": "system",
            "history_limit": 2,
            "max_context_chars": 100,
            "max_context_message_chars": 100,
        },
    )

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "older"},
        {"role": "assistant", "content": "recent"},
        {"role": "user", "content": "latest"},
    ]


def test_chat_templates_compile_with_interactive_forms():
    templates = Environment(loader=FileSystemLoader("sis/templates"))
    templates.filters["short_dt"] = str
    templates.filters["datetimeformat"] = str

    assert templates.get_template("chats/list.html")
    assert templates.get_template("chats/detail.html")

    sis_templates = SISTemplates(directory="sis/templates")
    request = SimpleNamespace(
        state=SimpleNamespace(username="local", csrf_token="csrf-token"),
        url_for=lambda *_args, **_kwargs: "/static/favicon.png",
    )
    rendered = sis_templates.env.get_template("chats/detail.html").render(
        request=request,
        conversation={
            "id": 1,
            "title": "HTML escaping",
            "provider": "stephanie",
            "created_at": "now",
            "updated_at": "now",
            "messages": [{
                "id": 1,
                "role": "assistant",
                "text": "<script>alert(1)</script>",
                "created_at": "now",
                "meta": {},
            }],
            "turns": [],
        },
        chat_status={"model_name": "fixture/model", "message": "disabled", "enabled": False},
        worker_status={"configured": False, "dry_run_only": False},
        worker_executions=[],
    )
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<script>alert(1)</script>" not in rendered


def test_database_url_and_direct_connection_share_structured_hydra_config():
    config = {
        "name": "stephanie_test",
        "user": "test_user",
        "password": "p@ss/word",
        "host": "db.example.test",
        "port": "5432",
    }

    url = database_url_from_config(config)
    connection = database_connection_parameters(config)

    assert url.database == "stephanie_test"
    assert url.username == "test_user"
    assert url.password == "p@ss/word"
    assert url.port == 5432
    assert connection["dbname"] == url.database
    assert connection["user"] == url.username
    assert connection["password"] == url.password
    assert "p@ss/word" not in url.render_as_string(hide_password=True)


def test_database_engine_factory_supports_isolated_sqlite_configuration():
    engine = create_database_engine(
        {"url": "sqlite://"},
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT 1").scalar_one() == 1
    finally:
        engine.dispose()


def test_database_engine_from_sis_hydra_config_uses_the_composed_target():
    engine = create_database_engine_from_hydra("sis")
    try:
        assert engine.url.database == "co"
        assert engine.url.username == "co"
    finally:
        engine.dispose()


def test_database_hydra_override_and_missing_configuration():
    engine = create_database_engine_from_hydra(
        "sis", overrides=["db.name=isolated_test"]
    )
    try:
        assert engine.url.database == "isolated_test"
    finally:
        engine.dispose()

    with pytest.raises(ValueError, match="name.*database"):
        database_url_from_config({})


def test_legacy_orm_aliases_require_explicit_config_and_release_engine():
    dispose_legacy_orm()
    engine = configure_legacy_orm({"url": "sqlite://"})
    try:
        with SessionLocal() as session:
            assert session.get_bind() is engine
        with pytest.raises(RuntimeError, match="different database"):
            configure_legacy_orm({"url": "sqlite:///another.sqlite"})
    finally:
        dispose_legacy_orm()


def test_memory_tool_passes_flat_hydra_bus_config_without_connecting():
    memory = object.__new__(MemoryTool)
    memory.cfg = {"bus": {"enabled": False, "backend": "inproc"}}
    memory.logger = FakeLogger()

    bus = memory._setup_knowledge_bus()

    assert bus.cfg == {"enabled": False, "backend": "inproc"}


def test_sis_chat_persists_user_and_assistant_messages_and_turn():
    app = make_app()
    with TestClient(app) as client:
        response = client.post("/chats/17/messages", json={"text": "New question"})

    assert response.status_code == 200
    result = response.json()
    store = app.state.memory.chats
    assert result["assistant_message"]["text"] == "A verified test response"
    assert [m.role for m in store.messages[-2:]] == ["user", "assistant"]
    assert [m.order_index for m in store.messages[-2:]] == [3, 4]
    assert store.messages[-1].meta["runtime"] == "stephanie-llm"
    assert store.messages[-1].meta["model"] == "test/model"
    assert store.messages[-1].meta["execution_status"] == "completed"
    assert store.messages[-1].meta["sis_request_id"]
    assert len(store.turns) == 1
    assert app.state.container.llm.messages[-1] == {
        "role": "user",
        "content": "New question",
    }
    assert all(message["role"] != "tool" for message in app.state.container.llm.messages)


def test_sis_chat_creates_conversation_in_existing_store():
    app = make_app()
    with TestClient(app) as client:
        response = client.post("/chats", json={"title": "Research notes"})

    assert response.status_code == 201
    assert response.json()["conversation"] == {"id": 18, "title": "Research notes"}
    assert app.state.memory.chats.created_conversation == {
        "provider": "stephanie",
        "title": "Research notes",
        "meta": {
            "source": "sis_live",
            "sis_owner": "local",
            "sis_create_request_id": response.json()["request_id"],
        },
    }


def test_interactive_chat_is_disabled_by_default():
    app = make_app(enabled=False)
    with TestClient(app) as client:
        response = client.post("/chats/17/messages", json={"text": "No send"})

    assert response.status_code == 404
    assert len(app.state.memory.chats.messages) == 3


def test_conversation_execution_rejects_another_local_owner():
    app = make_app()
    app.state.memory.chats.conversation.meta = {"sis_owner": "different-account"}
    with TestClient(app) as client:
        response = client.post("/chats/17/messages", json={"text": "Do not read this"})
        detail = client.get("/chats/17")

    assert response.status_code == 404
    assert detail.status_code == 404
    assert len(app.state.memory.chats.messages) == 3
    assert app.state.container.llm.messages is None


def test_interactive_chat_requires_model_and_live_persistence_gates():
    app = make_app()
    app.state.cfg["chat_runtime"]["readiness"]["model_verified"] = False
    with TestClient(app) as client:
        response = client.post("/chats/17/messages", json={"text": "Not ready"})

    assert response.status_code == 503
    assert "model" in response.json()["detail"]
    assert len(app.state.memory.chats.messages) == 3


def test_default_and_all_service_profiles_construct_existing_llm_service():
    memory = SimpleNamespace(bus=None)
    logger = FakeLogger()
    cfg = {"symbolic": {"enabled": False}, "bus": {}}

    for profile in ("default", "all"):
        container = ServiceContainer(cfg=cfg, memory=memory)
        load_services_profile(
            container,
            cfg=cfg,
            memory=memory,
            logger=logger,
            profile_path=f"config/services/{profile}.yaml",
        )
        llm = container.get("llm")
        assert llm.name == "llm"
        assert llm.container is container


def test_service_container_accepts_supervisor_logger_and_sync_lifecycle():
    memory = SimpleNamespace(bus=None)
    container = ServiceContainer(cfg={}, memory=memory, logger=FakeLogger())
    service = LifecycleService()
    container.register("test", lambda: service)

    assert container.get("test") is service
    assert container.get_service("test") is service
    assert container.get_or_none("missing") is None
    assert service.initialized
    container.shutdown_all()
    assert service.stopped


def test_service_container_requires_async_accessor_for_async_lifecycle():
    memory = SimpleNamespace(bus=None)
    container = ServiceContainer(cfg={}, memory=memory, logger=FakeLogger())
    service = LifecycleService(is_async=True)
    container.register("test", lambda: service)

    with pytest.raises(RuntimeError, match="get_async"):
        container.get("test")

    async def exercise_async_lifecycle():
        assert await container.get_async("test") is service
        assert service.initialized
        await container.shutdown()

    asyncio.run(exercise_async_lifecycle())
    assert service.stopped


def test_supervisor_resolves_the_hydra_service_profile_name():
    cfg = OmegaConf.create({"services": {"profile": "all"}})

    assert Supervisor._resolve_services_profile_path(None, cfg) == (
        "config/services/all.yaml"
    )


def test_secret_redaction_masks_credentials_but_keeps_safe_endpoints():
    redacted = redact_secrets(
        {
            "db": {"password": "db-password", "name": "database"},
            "model": {
                "api_key": "provider-key",
                "api_base": "http://localhost:11434",
                "input_tokens": 12,
            },
            "database_url": "postgresql://user:password@host/database",
        }
    )

    assert redacted["db"]["password"] == "[REDACTED]"
    assert redacted["model"]["api_key"] == "[REDACTED]"
    assert redacted["model"]["api_base"] == "http://localhost:11434"
    assert redacted["model"]["input_tokens"] == 12
    assert redacted["database_url"] == "[REDACTED]"
    error_text = redact_secrets(
        "connection failed postgres://user:db-pass@localhost/db api_key=provider-token"
    )
    assert "db-pass" not in error_text
    assert "provider-token" not in error_text


def test_plan_trace_persists_completion_when_scoring_is_disabled():
    class TraceStore:
        def __init__(self):
            self.saved = []

        def upsert(self, trace):
            self.saved.append(trace)

    traces = TraceStore()
    memory = SimpleNamespace(plan_traces=traces)
    service = PlanTraceService(
        cfg={
            "plan_monitor": {
                "enabled": True,
                "score_enabled": False,
                "index_embeddings": False,
            }
        },
        memory=memory,
        container=None,
        logger=FakeLogger(),
    )
    service.start_pipeline(
        {"goal": {"id": 1, "goal_text": "trace test"}, "pipeline": ["loader"]},
        "run-1",
    )

    asyncio.run(service.complete_pipeline({"final_output": "loaded 1 item"}))
    asyncio.run(service.score_pipeline({}))

    assert service.current_plan_trace.status == "completed"
    assert service.current_plan_trace.meta["status"] == "completed"
    assert traces.saved[-1].trace_id == "run-1"


def test_plan_trace_storage_failure_is_not_reported_as_pipeline_success():
    class FailingTraceStore:
        def __init__(self):
            self.calls = 0

        def upsert(self, _trace):
            self.calls += 1
            if self.calls > 1:
                raise OSError("database unavailable")

    traces = FailingTraceStore()

    service = PlanTraceService(
        cfg={
            "plan_monitor": {
                "enabled": True,
                "score_enabled": False,
                "index_embeddings": False,
            }
        },
        memory=SimpleNamespace(plan_traces=traces),
        container=None,
        logger=FakeLogger(),
    )
    service.start_pipeline(
        {"goal": {"id": 1, "goal_text": "trace failure test"}, "pipeline": []},
        "run-failure",
    )

    with pytest.raises(RuntimeError, match="Could not persist PlanTrace"):
        asyncio.run(service.complete_pipeline({"final_output": "not persisted"}))


def test_provider_failure_preserves_user_message_but_not_fake_assistant_reply():
    app = make_app(failure=True)
    with TestClient(app) as client:
        response = client.post("/chats/17/messages", json={"text": "Save my question"})

    assert response.status_code == 502
    assert "remains saved" in response.json()["detail"]["message"]
    assert [message.role for message in app.state.memory.chats.messages[-1:]] == ["user"]
    assert app.state.memory.chats.messages[-1].meta["execution_status"] == "failed"
    assert not app.state.memory.chats.turns


def test_context_budget_rejection_does_not_persist_a_message():
    app = make_app()
    app.state.cfg["chat_runtime"]["max_context_chars"] = 10
    with TestClient(app) as client:
        response = client.post("/chats/17/messages", json={"text": "too large"})

    assert response.status_code == 413
    assert len(app.state.memory.chats.messages) == 3


def test_chat_store_appends_with_monotonic_message_order():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ChatConversationORM.__table__.create(engine)
    ChatMessageORM.__table__.create(engine)
    ChatTurnORM.__table__.create(engine)
    store = ChatStore(sessionmaker(bind=engine, expire_on_commit=False))

    conversation = store.add_conversation(
        {"provider": "stephanie", "title": "Append test"}
    )
    first = store.add_messages(
        conversation.id,
        [
            {"role": "user", "text": "First"},
            {"role": "assistant", "text": "Second"},
        ],
    )
    second = store.add_messages(
        conversation.id,
        [{"role": "user", "text": "Third"}],
    )

    assert [message.order_index for message in first + second] == [0, 1, 2]
    assert [message.text for message in store.get_messages(conversation.id)] == [
        "First",
        "Second",
        "Third",
    ]
    engine.dispose()


def test_chat_store_reads_legacy_ids_turn_ratings_annotations_and_metadata():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ChatConversationORM.__table__.create(engine)
    ChatMessageORM.__table__.create(engine)
    ChatTurnORM.__table__.create(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory.begin() as session:
        conversation = ChatConversationORM(
            id=77,
            provider="legacy-provider",
            external_id="external-legacy-77",
            title="Imported legacy thread",
            meta={"source": "archive", "custom": {"retained": True}},
            tags=["legacy", "reviewed"],
        )
        user_message = ChatMessageORM(
            id=501,
            conversation_id=77,
            role="user",
            text="Legacy question",
            order_index=4,
            meta={"annotation": "user-origin"},
        )
        assistant_message = ChatMessageORM(
            id=502,
            conversation_id=77,
            role="assistant",
            text="Legacy answer",
            order_index=5,
            meta={"provider_id": "legacy-model-id", "temperature": 0.2},
        )
        turn = ChatTurnORM(
            id=91,
            conversation_id=77,
            user_message_id=501,
            assistant_message_id=502,
            order_index=5,
            star=4,
            ai_knowledge_score=87,
            ai_knowledge_rationale="Legacy annotation remains readable",
            ner=[{"text": "Stephanie", "label": "ORG"}],
            domains=[{"domain": "systems", "score": 0.91}],
        )
        session.add_all([conversation, user_message, assistant_message, turn])

    store = ChatStore(factory)
    try:
        conversation = store.get_conversation(77)
        messages = store.get_messages(77)
        turns = store.get_turns_for_conversation(77)

        assert conversation.id == 77
        assert conversation.provider == "legacy-provider"
        assert conversation.external_id == "external-legacy-77"
        assert conversation.meta["custom"]["retained"] is True
        assert conversation.tags == ["legacy", "reviewed"]
        assert [(message.id, message.order_index) for message in messages] == [
            (501, 4),
            (502, 5),
        ]
        assert messages[1].meta["provider_id"] == "legacy-model-id"
        assert len(turns) == 1
        assert turns[0].id == 91
        assert turns[0].star == 4
        assert turns[0].ai_knowledge_score == 87
        assert turns[0].ner[0]["text"] == "Stephanie"
        assert turns[0].domains[0]["score"] == 0.91

        appended = store.add_messages(77, [{"role": "user", "text": "Continue"}])
        assert appended[0].order_index == 6
        assert store.get_conversation(77).id == 77
    finally:
        engine.dispose()


def test_conversation_execution_uses_real_chatstore_atomic_turns_and_idempotency():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ChatConversationORM.__table__.create(engine)
    ChatMessageORM.__table__.create(engine)
    ChatTurnORM.__table__.create(engine)
    store = ChatStore(sessionmaker(bind=engine, expire_on_commit=False))
    create_payload = {
        "provider": "stephanie",
        "title": "Execution integration",
        "meta": {"sis_owner": "local"},
    }
    conversation = store.add_conversation_once(create_payload, "create-request")
    assert store.add_conversation_once(create_payload, "create-request").id == conversation.id
    llm = FakeLLM()
    service = ConversationExecutionService(
        memory=SimpleNamespace(chats=store),
        container=FakeContainer(llm),
        cfg={
            "chat_runtime": {
                "history_limit": 4,
                "max_context_chars": 2000,
                "max_context_message_chars": 500,
                "system_prompt": "Be concise and honest.",
                "model": {"name": "fixture/model", "api_base": "http://unused"},
            }
        },
        logger=FakeLogger(),
    )
    try:
        async def exercise():
            first = await service.execute_turn(
                conversation_id=conversation.id,
                username="local",
                text="First question",
                request_id="req-one",
            )
            replay = await service.execute_turn(
                conversation_id=conversation.id,
                username="local",
                text="First question",
                request_id="req-one",
            )
            assert replay["replayed"] is True
            assert replay["assistant_message"]["id"] == first["assistant_message"]["id"]
            assert len(store.get_messages(conversation.id)) == 2
            assert len(store.get_turns_for_conversation(conversation.id)) == 1

            second = await service.execute_turn(
                conversation_id=conversation.id,
                username="local",
                text="Follow-up question",
                request_id="req-two",
            )
            assert [message.role for message in store.get_messages(conversation.id)] == [
                "user", "assistant", "user", "assistant"
            ]
            assert len(store.get_turns_for_conversation(conversation.id)) == 2
            assert second["assistant_message"]["meta"]["model"] == "test/model"
            assert "provider" not in second["assistant_message"]["meta"]
            assert llm.messages[-3:] == [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "A verified test response"},
                {"role": "user", "content": "Follow-up question"},
            ]

            with pytest.raises(IdempotencyConflict):
                await service.execute_turn(
                    conversation_id=conversation.id,
                    username="local",
                    text="Different text",
                    request_id="req-one",
                )

        asyncio.run(exercise())
    finally:
        engine.dispose()


def test_failed_inference_can_retry_same_request_without_duplicate_user_message():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ChatConversationORM.__table__.create(engine)
    ChatMessageORM.__table__.create(engine)
    ChatTurnORM.__table__.create(engine)
    store = ChatStore(sessionmaker(bind=engine, expire_on_commit=False))
    conversation = store.add_conversation(
        {"provider": "stephanie", "title": "Retry test", "meta": {"sis_owner": "local"}}
    )
    llm = FakeLLM(failure=True)
    service = ConversationExecutionService(
        memory=SimpleNamespace(chats=store),
        container=FakeContainer(llm),
        cfg={
            "chat_runtime": {
                "history_limit": 4,
                "max_context_chars": 2000,
                "max_context_message_chars": 500,
                "system_prompt": "Be honest.",
                "model": {"name": "fixture/model", "api_base": "http://unused"},
            }
        },
        logger=FakeLogger(),
    )
    try:
        async def exercise_retry():
            with pytest.raises(ChatInferenceFailed):
                await service.execute_turn(
                    conversation_id=conversation.id,
                    username="local",
                    text="Keep this question",
                    request_id="retry-request",
                )
            assert len(store.get_messages(conversation.id)) == 1
            assert store.get_messages(conversation.id)[0].meta["execution_status"] == "failed"
            assert store.get_turns_for_conversation(conversation.id) == []

            llm.failure = False
            result = await service.execute_turn(
                conversation_id=conversation.id,
                username="local",
                text="Keep this question",
                request_id="retry-request",
            )
            assert result["assistant_message"]["text"] == "A verified test response"
            assert [item.role for item in store.get_messages(conversation.id)] == ["user", "assistant"]
            assert len(store.get_turns_for_conversation(conversation.id)) == 1

        asyncio.run(exercise_retry())
    finally:
        engine.dispose()


def test_active_chat_request_can_be_cancelled_without_creating_an_assistant_turn():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ChatConversationORM.__table__.create(engine)
    ChatMessageORM.__table__.create(engine)
    ChatTurnORM.__table__.create(engine)
    store = ChatStore(sessionmaker(bind=engine, expire_on_commit=False))
    conversation = store.add_conversation(
        {"provider": "stephanie", "title": "Cancellation", "meta": {"sis_owner": "local"}}
    )

    class BlockingLLM:
        def __init__(self):
            self.started = asyncio.Event()
            self.never_finish = asyncio.Event()

        async def achat(self, *_args, **_kwargs):
            self.started.set()
            await self.never_finish.wait()
            return {"text": "Should never be persisted"}

    llm = BlockingLLM()
    service = ConversationExecutionService(
        memory=SimpleNamespace(chats=store),
        container=FakeContainer(llm),
        cfg={"chat_runtime": {"history_limit": 4, "model": {"timeout": 30}}},
        logger=FakeLogger(),
    )

    async def exercise_cancel():
        task = asyncio.create_task(service.execute_turn(
            conversation_id=conversation.id,
            username="local",
            text="Cancel this request",
            request_id="cancel-request",
        ))
        await llm.started.wait()
        assert await service.cancel_turn(
            conversation_id=conversation.id,
            username="local",
            request_id="cancel-request",
        )
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(exercise_cancel())
        messages = store.get_messages(conversation.id)
        assert [message.role for message in messages] == ["user"]
        assert messages[0].meta["execution_status"] == "cancelled"
        assert store.get_turns_for_conversation(conversation.id) == []
    finally:
        engine.dispose()


def test_authenticated_chat_cancel_route_marks_request_cancelled_without_fake_turn():
    class BlockingLLM:
        def __init__(self):
            self.started = asyncio.Event()
            self.never_finish = asyncio.Event()

        async def achat(self, *_args, **_kwargs):
            self.started.set()
            await self.never_finish.wait()
            return {"text": "not expected"}

    llm = BlockingLLM()
    app = make_app(llm=llm)

    async def exercise_routes():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            request_id = "http-cancel-request"
            send = asyncio.create_task(client.post(
                "/chats/17/messages",
                json={"text": "Please cancel"},
                headers={"Idempotency-Key": request_id},
            ))
            await llm.started.wait()
            cancelled = await client.post(
                f"/chats/17/requests/{request_id}/cancel"
            )
            send_response = await send
        return cancelled, send_response

    cancelled, send_response = asyncio.run(exercise_routes())
    assert cancelled.status_code == 200
    assert cancelled.json()["cancelled"] is True
    assert send_response.status_code == 409
    assert "cancelled" in send_response.json()["detail"]["message"].lower()
    cancelled_messages = [
        item for item in app.state.memory.chats.messages
        if (item.meta or {}).get("sis_request_id") == "http-cancel-request"
    ]
    assert len(cancelled_messages) == 1
    assert cancelled_messages[0].meta["execution_status"] == "cancelled"
    assert not any(item.role == "assistant" for item in cancelled_messages)
    assert app.state.memory.chats.turns == []


def test_two_simultaneous_turns_for_one_conversation_are_serialized():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ChatConversationORM.__table__.create(engine)
    ChatMessageORM.__table__.create(engine)
    ChatTurnORM.__table__.create(engine)
    store = ChatStore(sessionmaker(bind=engine, expire_on_commit=False))
    conversation = store.add_conversation(
        {"provider": "stephanie", "title": "Concurrency", "meta": {"sis_owner": "local"}}
    )

    class SequencedLLM:
        def __init__(self):
            self.started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.calls = []

        async def achat(self, messages, **_kwargs):
            self.calls.append(messages)
            if len(self.calls) == 1:
                self.started.set()
                await self.release_first.wait()
            return {
                "text": f"response-{len(self.calls)}",
                "model_name": "fixture/model",
                "provider": "fixture",
            }

    llm = SequencedLLM()
    service = ConversationExecutionService(
        memory=SimpleNamespace(chats=store),
        container=FakeContainer(llm),
        cfg={
            "chat_runtime": {
                "history_limit": 8,
                "max_context_chars": 2000,
                "max_context_message_chars": 500,
                "system_prompt": "Use context in order.",
                "model": {"name": "fixture/model", "api_base": "http://unused"},
            }
        },
        logger=FakeLogger(),
    )

    async def exercise_concurrent_turns():
        first = asyncio.create_task(service.execute_turn(
            conversation_id=conversation.id,
            username="local",
            text="first",
            request_id="concurrent-one",
        ))
        await llm.started.wait()
        second = asyncio.create_task(service.execute_turn(
            conversation_id=conversation.id,
            username="local",
            text="second",
            request_id="concurrent-two",
        ))
        await asyncio.sleep(0)
        llm.release_first.set()
        return await asyncio.gather(first, second)

    try:
        results = asyncio.run(exercise_concurrent_turns())
        messages = store.get_messages(conversation.id)
        turns = store.get_turns_for_conversation(conversation.id)
        assert [message.role for message in messages] == [
            "user", "assistant", "user", "assistant"
        ]
        assert [turn.order_index for turn in turns] == [1, 3]
        assert [message["content"] for message in llm.calls[1][-3:]] == [
            "first", "response-1", "second"
        ]
        assert len(results) == 2
    finally:
        engine.dispose()


def test_llm_service_returns_only_observed_model_provenance(monkeypatch):
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return {
            "id": "response-42",
            "model": "qwen3:latest",
            "choices": [{"message": {"content": "A real-shaped fixture response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16},
        }

    monkeypatch.setattr("stephanie.services.llm_service.litellm.completion", completion)

    async def async_completion(**kwargs):
        captured.update(kwargs)
        return {
            "id": "async-response-43",
            "model": "qwen3:latest",
            "choices": [{"message": {"content": "Async fixture response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }

    monkeypatch.setattr("stephanie.services.llm_service.litellm.acompletion", async_completion)
    service = LLMService(
        cfg={},
        memory=SimpleNamespace(),
        container=FakeContainer(FakeLLM()),
        logger=FakeLogger(),
    )
    result = service.chat(
        [{"role": "user", "content": "hello"}],
        context={"conversation_id": 7},
        agent_cfg={
            "model": {
                "name": "ollama/qwen3",
                "api_base": "http://localhost:11434",
                "api_key": "test-secret",
                "timeout": 11,
                "max_tokens": 30,
            }
        },
        agent_name="sis_chat",
        add_to_history=False,
    )

    assert captured["timeout"] == 11
    assert captured["max_tokens"] == 30
    assert result["model_name"] == "qwen3:latest"
    assert result["provider"] == "ollama"
    assert result["response_id"] == "response-42"
    assert result["usage"] == {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16}
    assert result["finish_reason"] == "stop"
    assert result["latency_ms"] >= 0
    assert "test-secret" not in repr(result)

    async_result = asyncio.run(service.achat(
        [{"role": "user", "content": "hello"}],
        context={"conversation_id": 7},
        agent_cfg={
            "model": {
                "name": "ollama/qwen3",
                "api_base": "http://localhost:11434",
                "timeout": 11,
                "max_tokens": 30,
            }
        },
        agent_name="sis_chat",
        add_to_history=False,
    ))
    assert async_result["text"] == "Async fixture response"
    assert async_result["response_id"] == "async-response-43"
    assert captured["max_tokens"] == 30
