# Stephanie Next — Architecture Inventory (STN00 baseline / STN01 update)

**Audit date:** 2026-10-16  
**Repository:** `C:\Project\stephanie`  
**Branch / revision:** `elm` / `7584f47c9ef8acb5ee25f7b8bcd17c07d4999eee`  
**Initial worktree:** clean. No database writes or schema changes were made during the audit.

This inventory records the STN00 snapshot plus the STN01 delta below. Rows in the original inventory describe the STN00 baseline unless the STN01 section says otherwise. It is not a claim that every component is operational. “Verified” means only the named check.

## Executive summary

Stephanie already has a substantial Hydra-driven research runtime, a shared service container, SQLAlchemy memory stores, SIS, persistent conversations/messages/turns, PlanTrace, MemCube, casebooks, experiments, metrics, evaluation, and many learning/scoring agents. STN00 added an opt-in SIS chat path; STN02 now protects the application and verifies a real provider response through existing ChatStore/LLMService. The checked-in feature flag remains off until an operator provisions persistent local credentials and deliberately enables the verified gates.

The most important STN00 runtime issue was a database split: Hydra selected `co`, but SQLAlchemy sessions used a hard-coded `verity` URL. STN01 replaced the core runtime path with an engine built from the composed Hydra `db` node and verified ORM and direct connections against `co`. `verity` was not modified. The database password is now supplied through `STEPHANIE_DB_PASSWORD` rather than versioned YAML.

## Inventory

| Capability | Actual implementation | Status / evidence | Reuse decision | Gap / risk |
|---|---|---|---|---|
| Hydra entry point | `stephanie/main.py`, `@hydra.main(config_path="../config", config_name="config")` | `python -m stephanie.main --help` displays Hydra help; a `main()` wrapper now matches `pyproject.toml` | Keep | Full model-backed default pipeline was not run |
| SIS entry point | `sis/__main__.py`, `sis/main.py`, `sis/security.py` | Loopback default; all mounted HTTP routes require auth; host/origin/CSRF/session checks tested; readiness route exposes only booleans | Keep SIS as primary UI | Account credentials are not configured in current environment; remote HTTP is rejected and remote HTTPS requires explicit allowlists |
| Hydra workflow lifecycle | `stephanie/supervisor.py`: parse configured stages, create pipeline-run record, run dynamically imported agents, update PlanTrace, report outcomes | Repeatable `scripts/stn01_smoke.py --execute` runs existing `LoaderAgent`; run `10026` and its completed one-step PlanTrace were retrieved after reopening MemoryTool | Keep Hydra and Supervisor | Full default/model-backed pipeline remains unverified |
| Agent loading | `Supervisor._run_pipeline_stages` resolves each configured `stage.cls` through `hydra.utils.get_class`; separate registries also exist | One existing `LoaderAgent` stage executed through the Hydra-composed Supervisor | Keep dynamic Hydra stage loading | There are multiple registration patterns (`hydra.utils.get_class`, `AgentRegistry`, component registries); clarify ownership before introducing another registry |
| Service container | `stephanie/services/service_container.py`, profile loader in `registry_loader.py`, profiles under `config/services/` | Default/all profiles construct; async service init/shutdown is awaited through explicit async APIs; container uses MemoryTool's bus; legacy `get_service()` alias retained | Keep | `ChatCorpusService` expects a `chat_store` service that is not registered by the active profiles; some older consumers of async services need conversion to `get_async()` |
| Four foundations | `docs/design.md`: Configuration, append-only data Log, MemoryTool, Container | Read and preserved; runtime DB identity now derives from Hydra | Preserve all four | Standalone shadow persistence uses an explicit DSN; no generic migration ledger yet |
| Configuration | Hydra/OmegaConf YAML in `config/`; SIS composes `config/sis.yaml`; research entry point composes `config/config.yaml` | `config`, `sis`, and `chat` composition passed using `venv\Scripts\python.exe` | Keep Hydra | Default Python on PATH is not the repository venv and lacks Hydra/OmegaConf |
| Structured logging | `JSONLogger`, `core/logging`, JSONL paths in `logs/` | STN01 redacts config/run/agent/PlanTrace diagnostics and PACS artifact config; LLM errors log type only | Reuse | Other legacy exception logs still need audit |
| Memory / database | `MemoryTool`, `BaseSQLAlchemyStore`, many stores; PostgreSQL/pgvector dependencies | ORM and direct connection identity both report `co`; legacy chat and PlanTrace rows were read | Keep MemoryTool and existing stores | No migration chain; `verity` remains untouched. Schema creation is not used as an upgrade mechanism |
| Chat ORM | `ChatConversationORM`, `ChatMessageORM`, `ChatTurnORM` in `stephanie/orm/chat.py` | `co` has all three tables; legacy conversation and six messages were read through configured ChatStore | Keep stable IDs/tables | No live run/session schema; do not repurpose imported conversation IDs |
| Chat persistence | `stephanie/memory/chat_store.py` provides conversation/message/turn CRUD, import metadata, annotation, ratings, retrieval, idempotent prompt append and atomic assistant+turn commit | Legacy-compatible SQLite tests pass; real two-turn conversation `1665`, messages `83552`–`83555`, turns `37314`/`37315`, persisted in `co` and read in a fresh process | Keep ChatStore canonical | Seven clearly titled acceptance conversations were added; no old conversation records were changed |
| Chat import/learning | `ChatImportAgent`, `chat_importer.py`, `ChatCorpusService`, `LearningFromLearningAgent`, chat-to-CaseBook pipeline | Source/config present; not operationally demonstrated | Reuse import and learning pipeline as separate workflows | `ChatCorpusService` expects `ensure_thread` / `add_message` store methods absent from `ChatStore`; the service remains unregistered and its storage contract is stale |
| SIS chat routes | `sis/routes/chats.py`, `sis/services/chat_execution.py`, existing templates | Authenticated/CSRF-protected create/send/cancel exercised; real configured model and fresh-process readback passed | Reuse SIS routes and templates | Non-streaming; no tool execution or PlanTrace per chat; checked-in feature/readiness flags remain false |
| LLM/model support | Existing `LLMService.complete()`/`chat()` plus async `achat()` through LiteLLM and ServiceContainer | Real `ollama/qwen3` chat passed; actual provider/model/usage/latency recorded. Missing API key is omitted, not sent as empty Bearer header | Reuse LLMService; do not build another provider layer | No claims about other provider configurations; cancellation is best effort at async provider request |
| Embeddings / retrieval | `MemoryTool` chooses HNet/HF/other embedding stores; `ChatCorpusService` calls semantic turn search | Source/config verified, no embedding service health check | Reuse bounded retrieval later | Do not inject whole chat history; missing embedding backend must degrade clearly |
| PlanTrace | `PlanTraceService`, `PlanTraceStore`, `PlanTraceORM`, `ExecutionStepORM` | Run `10026` persisted and was re-read after closing/reopening MemoryTool; meta status `completed`, one step | Reuse for Hydra evidence | Not a generic chat/external-runtime run record; singleton `current_plan_trace` is unsafe for concurrent tasks |
| MemCube / knowledge | `memcube/`, `MemCubeStore`, lifecycle/factory, cartridges/belief graph; CaseBook stores | Source/config verified; no operational validation | Reuse, do not add a second memory/index | Distinguish transcript, observations, verified facts, skills and preferences in later policy/schema |
| Experiments and metrics | `ExperimentStore`, `MetricStore`, score/evaluation stores and experiment ORM models | Source verified; production data not inspected beyond table names | Extend existing experiment tracking | Candidate-artifact lineage, held-out protocol and promotion gates are not yet one coherent workflow |
| Learning and scoring | GILD trainer/signals, SICQL scorer/trainer, PolicyAnalyzer, LearningFromLearning, prompt tuning/compilation, model and strategy routing, scoring/calibration | Many implementations/configs exist; source presence only, not proven operational | Retain and integrate by explicit use cases | Avoid equating training/scoring agents with demonstrated self-improvement |
| Skills | PACS skill extraction/preparation/validation, filters, configs and skill reports | Source/config present; no artifact-to-held-out improvement demonstrated locally | Reuse after evaluation protocol exists | Skill classes are not yet a unified, evidence-backed reusable procedural store for external runtime runs |
| Buses / async | Hybrid NATS/in-process and ZMQ services; async Supervisor stages; Windows event-loop helper | NATS `localhost:4222` was unavailable; synchronous request/response chat uses no bus | Reuse existing bus where a real async need exists | No new event bus for chat; durable jobs and disconnect/restart recovery remain later work |
| Files / artifacts | `data/`, `outputs/`, `runs/`, `reports/`, `logs/`; several JSONL and report helpers | Directories exist; no retention policy audit completed | Reuse filesystem conventions | Retention, secret redaction and workspace isolation need explicit policy |
| Migration tooling | `sis/db.py:init_db()`, schema dump scripts and `schema_raw.sql` | No Alembic directory/config or ordered migration chain found; `init_db()` now requires explicit schema-creation opt-in | Establish a migration process before adding tables | A separate per-worldview SQLite locator still creates its own local schema; it was not invoked in STN01. `create_all` is not an upgrade plan for the canonical database |
| Pi runtime | Installed Pi `1.2.0`; `pi --help` documents `--mode rpc`, session flags, tool allowlists and JSON output | Binary availability and flags verified; no task launched | Use official JSONL RPC subprocess adapter; Pi retains its own agent loop | Must test process streaming/cancel/timeout and Windows pipes, in a bounded workspace |
| Worker gateway | Adjacent `C:\Projects\worker` checkout, package `0.2.0`, versioned `/v1/work` API | An existing `bun ... worker serve` process responded on `127.0.0.1:7317`; health reports `opencode`/`opencode-work` available and `/v1/jobs` was empty. Its response lacks the new listener/auth fields; no real Work request was submitted. The separate zero-price LongCat probe failed 503. | STN03 reuses Worker; no second OpenCode runtime in Stephanie | Optional bearer auth exists in checkout but is not enabled in active Worker; jobs/results are bounded in-memory; loopback does not isolate other local processes |
| OpenCode runtime | Installed OpenCode `v2.0.29`; basic Worker executor invokes `opencode run --standalone --format json` | GPT-6 Luna route `opencode-go/gpt-6-luna` was discovered as metered and passed one tools-denied direct inference check; observed usage/cost are recorded in `stn03-live-acceptance.md`. Configured DeepSeek route was absent. No Worker coding task or SIS OpenCode session was run. | Use Worker as the initial gateway; native API/SDK remains a later separate adapter | Basic Worker executor is not a session server or read-only sandbox; permissions, network and host filesystem need external isolation |
| Sidecar / Workbench | No integration located in the audited local paths | Unknown / not demonstrated | Optional later integration only | Separate product contracts and authorization boundaries must be established first |
| Tests | `tests/test_sis_worker_execution.py`, chat/security tests and existing suite | 58 combined SIS focused tests pass; narrowed suite has 117 passes and the same 3 external-fixture failures; Worker suite 211 pass/8 skipped and typecheck passes | Keep adding isolated chat/security tests | Full collection still aborts in `test_elm.py`; three portfolio tests need missing `C:\Projects\new-books` files |

## Chat path and data ownership

```text
Browser → SISAuthMiddleware → chats router → ConversationExecutionService
                                             ├─ MemoryTool.chats (ChatStore)
                                             └─ ServiceContainer.get("llm") → LLMService.chat → LiteLLM
```

The live path persists through existing conversation/message/turn tables. The importer and historical-analysis workflows continue to use these tables. A workflow/runtime run should receive an explicit STN03 identity later rather than replacing `ChatConversationORM.id` or treating an external-runtime session as a conversation.

`ChatCorpusService` is a distinct, partially stale “thread” facade; its current expected store API does not match `ChatStore`. Keep it out of the live path until the ownership mismatch is resolved.

## Database and migrations

- `config/db/postgres.yaml` selects database/user `co` (local, port 5432); the password is supplied by `STEPHANIE_DB_PASSWORD` and is not committed.
- `stephanie/orm/base.py` provides `database_url_from_config()` and `create_database_engine()`; no engine is created at import. `MemoryTool` binds ORM stores and the direct pgvector connection to that same Hydra node.
- `MemoryTool.database_identity()` reports safe configured, ORM and direct connection identities. STN01 verification reports all three as `co`, with matching schema/search path/user.
- Read-only schema inspection found chat, PlanTrace, MemCube and experiment tables in `co`. The legacy `verity` database was not modified.
- No Alembic configuration or ordered migration chain was found. No canonical database migration or `create_all()` was run; `sis.db.init_db()` now requires explicit schema-creation opt-in and is not called by SIS startup. A separate local SQLite worldview helper retains its own creation behavior.
- Standalone administrative and read-only viewer utilities now use Hydra's database config; data-changing tools default to dry-run. Destructive Nexus benchmarking requires separate scratch-database protections. Optional evaluation-shadow persistence and standalone visualization tools may still require an explicit DSN and pre-provisioned schema.

## Baseline health and reproducibility

| Check | Result |
|---|---|
| Current revision / branch | `7584f47c9ef8acb5ee25f7b8bcd17c07d4999eee` / `elm`; STN00 started clean; STN01 changes remain uncommitted |
| Hydra config composition | `config`, `sis`, and `chat` all composed using `venv\Scripts\python.exe` |
| SIS import | Succeeded with 22 routes after a >45-second first attempt; service profile now registers `llm`; model components loaded and GLiNER2 was unavailable, falling back to BERT NER |
| PostgreSQL | Windows PostgreSQL 16 service running; `co` read-only connection succeeded |
| SQLAlchemy/direct database identity | MemoryTool diagnostic reports configured/ORM/direct database `co`; equality passed |
| Legacy ChatStore | One existing conversation and its six messages read through MemoryTool; no message content printed or changed |
| PlanTrace persistence | Pipeline/trace `10026` found after MemoryTool closed and reopened; status `completed`, one execution step |
| Controlled Hydra runtime | `config/code.yaml` one-stage variant, existing `LoaderAgent`, one temporary JSONL input; run and trace persisted without model/bus calls |
| Ollama | `localhost:11434` unavailable at audit time; real model call not attempted |
| NATS | `localhost:4222` unavailable at audit time |
| Pi / OpenCode | Installed commands report Pi `1.2.0`, OpenCode `v2.0.29` |
| venv package health | `venv\Scripts\python.exe -m pip check`: no broken requirements; optional `test` dependency group installed pytest |
| PATH Python | Python 3.13 has pytest but lacks Hydra/OmegaConf; global `pip check` reports `compressed-tensors` requires torch >=2.10 while PATH Python has torch 2.6.0+cu124 |
| Baseline test collection | `python -m pytest --collect-only -q` aborts while importing `tests/test_elm.py`: missing `AxisDirection`, then `CalibratedThresholds` NameError/SystemExit |
| Baseline tests excluding `test_elm.py` | 59 passed, 3 failed before STN00/STN01 changes. All three portfolio-experiment cases expect `C:\Projects\new-books\content\books\dspy-from-first-principles\07-chapter.md`, absent in this environment |
| CLI/startup documentation | `docs/installation.md` corrected; README still refers to stale `schema.sql`/`run_pipeline.py` paths |

No real model call, schema migration, Pi task or OpenCode task was performed. The default multi-stage model-backed pipeline remains unverified; the controlled Hydra/ Supervisor variant did run and persist evidence. See `stn01-stabilization-results.md` for the STN01 command/trace details and an earlier incomplete smoke trace.

## STN00/STN01 baseline chat slice (implemented locally)

- Added opt-in SIS live-conversation create/send routes and a small UI composer, calling the existing `LLMService.chat()` and storing through `ChatStore`.
- Fixed service-profile construction of the existing LLM service by supplying its required container argument. Corrected `NerTool` to reuse the existing GLiNER2/BERT-fallback `EntityDetector`, removing a bad import that prevented profile registration.
- At STN00 the feature default was false and SIS lacked authentication. STN02 adds the whole-app access boundary and retains `chat_runtime.enabled: false` plus false model/persistence assertions until a persistent account is configured and activation is deliberate.
- Message history is bounded by configured count and character limits; only `user` and `assistant` messages enter model context. Tool/system transcript roles are excluded.
- STN01 saved user messages before inference. STN02 retains that behavior, adds idempotency/failure status and atomic assistant+turn persistence, and never invents an assistant response.
- Fixed `ChatStore.add_messages()` to allocate monotonic indices under a parent-conversation row lock and update `updated_at`.
- At STN01, focused tests covered SQLite, isolated services, Jinja compilation and a deterministic fake LLM. Those earlier tests did not verify live inference; current STN02 evidence follows.
- No schema migration was needed for this slice.

## STN02 delta — current chat/security state

- SIS uses a single configured local account, revocable process-local session, PBKDF2 password verifier, CSRF tokens and exact Host/Origin checks for all mounted application routes. `/healthz`, login and static assets are the only unauthenticated HTTP paths; docs/OpenAPI are disabled.
- SIS fails closed when account/session secrets are absent. `python -m sis` still binds to loopback. Remote binding requires allowlisted host/origin and HTTPS. One process/worker is required for in-memory sessions.
- The `ConversationExecutionService` now owns bounded context, per-conversation serialization, idempotency and model invocation. ChatStore writes the genuine assistant message and turn atomically; failed inference preserves the user message without a synthetic assistant/turn.
- Actual local multi-message `ollama/qwen3` inference passed through async LLMService and the authenticated SIS route. Acceptance conversation `1665` was read after a separate process reopened MemoryTool against `co`; model/provider/usage/latency are recorded in assistant metadata.
- `config/sis.yaml` intentionally remains `chat_runtime.enabled: false`, `readiness.model_verified: false`, and `readiness.live_persistence_verified: false`. The authenticated acceptance runner used an ephemeral test account and in-memory gate override; the local environment has no persistent SIS account configured.
- No schema migration was needed. Seven test-labeled acceptance conversations were added; existing legacy conversations remain unchanged.
- Combined tests: 38 focused passes; narrowed existing suite 97 passes and the same three external fixture failures. Full collection still fails in `test_elm.py`.

See [`acceptance-matrix.md`](acceptance-matrix.md) for specific pass/fail status and [`milestones.md`](milestones.md) for the revised dependency order.
