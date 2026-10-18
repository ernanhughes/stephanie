# STN01 — Stabilization Results

**Decision:** STN01 acceptance gates passed with the listed limits. Live SIS chat remains disabled.  
**Repository:** `C:\Project\stephanie` · branch `elm` · base revision `7584f47c9ef8acb5ee25f7b8bcd17c07d4999eee`  
**Database:** application path verified against configured database `co`; no migration or schema DDL was run.

## A. Root causes

### Database split

`MemoryTool` opened a raw psycopg2 connection from Hydra `cfg.db`, but built SQLAlchemy sessions from the module-level `engine` in `stephanie/orm/base.py`, whose URL was hard-coded to `verity`. This divided raw pgvector calls from every SQLAlchemy store, including ChatStore and PlanTraceStore. A read-only inspection found the complete chat and PlanTrace tables in `co`, while `verity` lacked `chat_conversations`.

### Runtime/service construction

- `Supervisor` passed `logger=` to `ServiceContainer`, whose constructor did not accept it.
- `ServiceContainer.initialize()` constructed a second bus and referenced a missing logger; its async shutdown awaited synchronous service shutdowns.
- Synchronous `get()` called async service `initialize()` without awaiting it.
- Supervisor ignored the selected Hydra `services` group and defaulted to `services/default`.
- A stale NER import prevented profile registration, and the LLM service profiles omitted the required container argument.

### Trace lifecycle

An STN01 smoke override set `score_enabled=false`. A misplaced guard initially skipped `complete_pipeline()` as well as scoring, leaving a trace in progress without its stage step. The guard was corrected; a regression test verifies trace completion still persists when scoring is disabled.

## B. Changes implemented

- `stephanie/orm/base.py`: added structured Hydra DB URL/engine construction; removed the hard-coded `verity` target and all import-time engine creation. Legacy `engine`/`SessionLocal` aliases require an explicit `configure_legacy_orm(cfg.db)` call and can be disposed explicitly.
- `stephanie/memory/memory_tool.py`: binds SQLAlchemy sessions and the direct pgvector connection from the same `cfg.db`; adds redacted identity diagnostics and resource cleanup.
- `stephanie/memory/base_store.py`, `stephanie/utils/db_scope.py`, `stephanie/utils/db_retry.py`: dispose the pool belonging to the operation rather than a hard-coded global engine.
- `stephanie/services/service_container.py`: accepts the Supervisor logger, reuses MemoryTool's bus, preserves dependency order, detects cycles, provides `get_async()` for async service lifecycle, and awaits sync/async shutdown correctly.
- `stephanie/supervisor.py`: honors the Hydra service-profile choice and stores redacted run configuration.
- `stephanie/services/plan_trace_service.py`: supports disabling scorer/embedding work independently of trace persistence; records completed/incomplete/failed state in trace metadata.
- `stephanie/utils/secret_utils.py`, `stephanie/agents/base_agent.py`, `stephanie/scoring/training/pacs_trainer.py`, `stephanie/services/llm_service.py`, `stephanie/services/plan_trace_service.py`, `stephanie/supervisor.py`, `stephanie/main.py`: redact sensitive config/error values in diagnostics, traces and artifacts; avoid logging raw provider exception text.
- `stephanie/main.py`, `sis/main.py`, `stephanie/memory/memory_tool.py`: close container, raw connection and ORM pool at application/CLI shutdown.
- `scripts/stn01_smoke.py`: repeatable model-free Hydra/Supervisor smoke; read-only by default and requires `--execute` before writing run evidence.
- `sis/__main__.py`: local Uvicorn binding defaults to loopback (`127.0.0.1`); remote binding requires explicit `SIS_HOST` and remains unsafe without authentication.
- `config/db/postgres.yaml`: DB password now comes from `STEPHANIE_DB_PASSWORD`; no credential is hard-coded there.
- Standalone DB maintenance/read-only utilities now use Hydra's `db` configuration; data-changing tools default to dry-run. The destructive Nexus benchmark requires a separate scratch DB DSN and an explicit destructive confirmation, and rejects `co`/`verity`. Optional evaluation-shadow and visualization paths still accept explicit DSNs.
- `sis/db.py:init_db()` no longer creates tables by default; local/test creation requires explicit `create_schema=True`.
- `docs/installation.md` corrected; `stephanie.main:main` now exists as the declared console-script entry point.
- STN00's interactive chat code remains opt-in with `chat_runtime.enabled: false`.

## C. Database and data safety

Read-only identity checks through MemoryTool reported:

```text
configured database = ORM database = direct connection database = co
identity match = true
```

Before the smoke writes, MemoryTool reported 1,641 conversations and 1,973 PlanTraces. The follow-up read used the normal path to retrieve one legacy conversation and its six messages, and a completed PlanTrace. Existing conversation identifiers/content were not changed. `verity` was not altered or migrated.

Read-back of run `10026` confirmed its stored `run_config.db.password` is `[REDACTED]`; Hydra `--help` did not display the resolved environment credential.

No `create_all`, migration, table drop, data copy, or schema write occurred. The controlled pipeline added a small number of goal/run/trace records to the configured local database, as required to prove persistence. One initial invocation failed before run insertion but after creating a unique smoke goal; it was left in place rather than deleted.

## D. Genuine Hydra workflow and PlanTrace evidence

The smoke was a **controlled one-stage variant of existing `config/code.yaml`**, using its existing `critic_loader`/`LoaderAgent`, Hydra composition, Supervisor, configured `services: all` profile, and a temporary one-record JSONL fixture. Overrides set `bus.enabled=false`, `plan_monitor.enabled=true`, `score_enabled=false`, `index_embeddings=false`, and disabled reports. No LLM, NATS, or embedding provider was called.

Accepted execution:

| Evidence | Result |
|---|---|
| Database identity | `co`, matched |
| Pipeline name | `stn01_loader_smoke` |
| PipelineRun DB ID / PlanTrace ID | `10026` |
| Pipeline execution UUID | `fe3afbe9-bd8f-4197-8bae-cbd9bba9a6f8` |
| Loaded fixture records | 1 |
| Stage | `critic_loader` (`LoaderAgent`) |
| Services resolved by the smoke | `plan_trace`, `rules`, `scoring`, `state` |
| Persisted PlanTrace state | `completed` |
| Persisted execution steps | 1 |
| Restart/read-back | Reopened MemoryTool in a separate process; run and trace were found |

An earlier diagnostic run (`10021`) is preserved as `incomplete`, with a note recording the initial trace-completion guard defect. Corrected runs `10022`–`10026` completed. No test evidence was represented as a model-backed success.

## E. Security and operations

- `python -m sis` defaults to `127.0.0.1`; no authentication middleware was added.
- Interactive chat remains disabled. Do not set a remote `SIS_HOST` or enable live chat without an explicit access-control boundary.
- Passwords are supplied through environment variables and redacted from saved run configuration and selected diagnostics.
- The repository venv's `pip check` is clean. Pytest is now declared in the optional `test` extra and installed in that venv. PATH Python lacks Hydra/OmegaConf and has a separate torch/compressed-tensors version conflict; use the venv interpreter.
- Ollama (`localhost:11434`) and NATS (`localhost:4222`) were unavailable. No provider call or NATS-backed workflow was attempted.

## F. Verification

| Check | Result |
|---|---|
| Focused STN01/STN00 tests | 22 passed |
| Existing suite, excluding `tests/test_elm.py` | 81 passed, 3 failed (missing external `C:\Projects\new-books\...\07-chapter.md`) |
| Full test collection | Still aborts in `tests/test_elm.py` on `AxisDirection` / `CalibratedThresholds` import failures |
| `venv\Scripts\python.exe -m pip check` | No broken requirements |
| Python compile check | Passed for changed Python modules |
| Hydra CLI | `venv\Scripts\python.exe -m stephanie.main --help` displayed Hydra help |
| Repeatable smoke helper | `venv\Scripts\python.exe scripts\stn01_smoke.py` performs no DB access; `--execute` run `10026` passed and was read back in a fresh process |
| SIS local host | Patched Uvicorn runner reports default `127.0.0.1:8000` |
| SIS app/service | App imported with 22 routes; LLM service registered and reports healthy; no completion call |
| Diff hygiene | `git diff --check` passed (Git reports expected LF→CRLF working-copy notices) |

Focused test command (PowerShell):

```powershell
venv\Scripts\python.exe -m pytest tests/test_sis_chat_execution.py -q
```

The venv test dependency can be installed with `venv\Scripts\python.exe -m pip install -e ".[test]"`.

Existing suite command:

```powershell
venv\Scripts\python.exe -m pytest --ignore=tests/test_elm.py -q
```

## G. STN01 acceptance

| Gate | Status |
|---|---|
| DB-01 Hydra/ORM/direct target match | **PASS** |
| DB-02 MemoryTool/ChatStore consistency | **PASS** |
| DB-03 PlanTraceStore target | **PASS** |
| DB-04 legacy conversation readability | **PASS** |
| DB-05 no schema changes | **PASS** |
| RT-01 Supervisor construction | **PASS** |
| RT-02 required service initialization | **PASS for selected pipeline/profile** |
| RT-03 configured Hydra workflow | **PASS for controlled existing LoaderAgent variant** |
| RT-04 persisted/retrieved PlanTrace | **PASS** |
| SEC-01 interactive chat disabled | **PASS** |
| SEC-02 local-only default binding | **PASS for `python -m sis`; remote auth remains absent** |
| TEST-01 focused regression tests | **PASS (22 tests)** |
| TEST-02 existing failures separately classified | **PASS; failures remain** |
| DOC-01 implementation docs | **PASS; README still has legacy commands** |

## H. Next milestone

STN01 is complete with limitations. STN02 may continue the SIS chat slice in deterministic/disabled mode. Do not enable live chat until a local access boundary and configured model are verified. First acceptance should create/resume/send through SIS on the canonical DB and read the persisted ordered conversation after closing and reopening the app.
