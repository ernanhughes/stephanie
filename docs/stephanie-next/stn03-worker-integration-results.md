# STN03 — Worker Integration Results

**Outcome:** The Worker-backed integration path is implemented and deterministically tested. **The exact GPT-6 Luna route passed one bounded direct OpenCode inference check, but end-to-end SIS coding acceptance remains BLOCKED; no live coding execution was claimed.** The SIS Worker route is explicitly set to `opencode-go/gpt-6-luna` and its local model verification flag reflects that probe; `worker.enabled` remains false, Worker API auth is required, and no token/workspace is provisioned. Worker's independent production model policies and the global OpenCode config were not changed. The pre-existing Worker service was left running untouched; no database migration or commit/push occurred.

## A. Architecture

SIS uses a small Python `requests` client against Worker's existing loopback HTTP API. A `WorkerExecutionService` is registered with Stephanie's existing `ServiceContainer`, reads Hydra `worker` configuration, and never starts Worker at SIS boot. The explicit chat action dispatches to a bounded thread pool. Worker still owns OpenCode invocation and status semantics.

## B. Implementation

- Added `sis/services/worker_execution.py`: loopback-only Worker client, health/executor/model discovery checks, versioned request/result validation, path registry, explicit backend/model selection, bounded dispatch, timeout/uncertain-outcome handling, result redaction, status mapping and evidence linking.
- Extended `stephanie/memory/chat_store.py` to persist idempotent execution/task links and bounded result metadata in existing conversation JSON. No table/schema migration. `sis/services/chat_context.py` marks bounded external Worker output with its provenance/status on follow-up turns.
- Added authenticated, CSRF-protected execution/readiness/status/evidence endpoints in `sis/routes/chats.py`; wired the service in `sis/main.py`; added the explicit chat-view control/polling and evidence page to SIS templates.
- Added a default-on-for-request `dry_run` option in SIS; a dry-run uses Worker `/health/transport` and returns an explicit `DRY RUN — NO AGENT EXECUTION` state. Worker core now records dry-run Work before executor health/execute, and returns without invoking OpenCode.
- Updated `stephanie/memory/plan_trace_store.py` to persist the already-modeled `ExecutionStep.step_type` field for external lifecycle steps.
- Added disabled-by-default Hydra `worker` configuration in `config/sis.yaml`.
- Changed Worker's basic OpenCode transport from blocking `spawnSync` execution to bounded asynchronous child-process execution with best-effort abort and output limits. Added opt-in bearer auth for Work submission and sensitive job/result/model operations; no secret is configured. No custom agent selection is exposed or sent. Worker `serve` now binds on loopback even if its server host config is wildcard/remote.
- Added `tests/test_sis_worker_execution.py` using a local fake HTTP Worker and a temporary SQLite ChatStore; added Worker executor and loopback-serving tests.

## C. Worker and model verification

- Worker checkout: `C:\Projects\worker`, package/health version `0.2.0`; checkout HEAD observed as `0aecefa95c5ccff25290179825ee6a1fde2f1ca5`. Worker already had unrelated user modifications in README/core/server/web/context files; they were preserved.
- OpenCode CLI: `opencode v2.0.29`.
- `bun src/cli/cli.ts health` reported both `opencode` and `opencode-work` executor health available in the locally constructed service; the basic executor reports the CLI version. The model-runtime route/quota state was `UNKNOWN`.
- An existing Worker HTTP service was listening at `127.0.0.1:7317` (OS listener inspection confirmed loopback, process command was `bun src/cli/cli.ts serve`, and `/v1/jobs` was empty). I reused it without restarting or starting a duplicate. Its health response lacks the new `listener`/`security` fields, so the updated SIS client correctly refuses this legacy-running instance; it was left untouched.
- `opencode auth list` showed stored OpenCode Go credentials, but this does not prove a usable model route or remaining authorization/quota. `opencode models` listed `opencode-go/deepseek-v4-pro` and `opencode-go/deepseek-v4.1-flash`, but not the existing user-configured `opencode-go/deepseek-v4-flash`. It listed `opencode/deepseek-v4-flash`, a different provider route; no provider substitution was made.
- Worker discovery returned 14 explicitly zero-priced CLI models, including `opencode-go/longcat-2.5-preview-free`. I ran exactly one requested bounded probe (`--max-requests 1`) through Worker's tools-denied temporary probe environment. It returned `FAILED`, classification `unavailable`, HTTP 503, 2.811 seconds; authentication and quota remain unknown. Worker recorded a local backoff until 21:25:31. I did not retry or probe another model.
- Worker CLI discovery (without `--free`) also confirms `opencode-go/gpt-6-luna` and marks it metered (`free: false`). Separately, the requested one-shot model check used OpenCode CLI directly, not Worker's zero-priced probe API: `opencode run --standalone --format json --model opencode-go/gpt-6-luna --title 'STN03-LIVE bounded metered probe' 'Reply exactly OK.'` from a unique temporary directory.
- That call **succeeded**. Sanitized OpenCode session export observed provider/model `opencode-go/gpt-6-luna` (variant `default`), session `ses_eb947dfaeffeV0UUrGLHxJLn61`, assistant message `msg_146b8242c001r18jELsjoL8qbg`, reply `OK`, finish `stop`, input tokens 770, output tokens 5, reasoning 0, cache read/write 0, reported cost `$0.0000795`, and elapsed session time 2,937 ms. The temporary project config was verified by `opencode debug config` to have a deny-all permission policy; the exported session contained only user/assistant/idle messages and no tool call. This metered OpenCode-only probe is not a Worker Work request, SIS conversation, or PlanTrace.
- Worker’s basic `opencode` backend was selected in the disabled config; `opencode-work` remains opt-in and was not executed.

## D. SIS demonstration

The deterministic tests exercise the actual SIS FastAPI route behind `SISAuthMiddleware` with a real test session, Origin and CSRF token, through `WorkerExecutionService` to a controlled local fake Worker HTTP service; TestClient also rendered the authenticated conversation and evidence pages. A separate direct OpenCode GPT-6 Luna model check succeeded, but no real Worker-backed response or browser submission was demonstrated. This checkout still has no configured SIS username/password verifier/session secret, and the chat readiness gates remain false. The Worker execution feature cannot be activated for a live user without operator provisioning and isolation.

## E. Evidence

The temporary-database acceptance verified a generated Stephanie execution ID linked to the fixture conversation, user task message, assistant message/ChatTurn, fake Worker request ID, fake OpenCode session reference and `worker:<execution ID>` PlanTrace reference. Execution metadata was read through a second ChatStore engine; the actual PlanTraceStore inserted/updated the trace in a temporary SQLite schema and a second engine reopened its trace and steps. The fixture adapted PlanTrace's unused PostgreSQL `ARRAY` column to SQLite `JSON` only inside the test; production ORM/schema were not changed. This was fixture evidence only: the fake Worker ID/session and temporary trace are **not** a live SIS/Worker execution, production database row, or real conversation. The separate direct OpenCode session/message IDs are recorded in section C, but they are not linked to SIS or PlanTrace.

## F. Security

- SIS execution routes inherit whole-app authentication, conversation ownership, Host/Origin and CSRF controls; the execution additionally depends on the existing chat/model/persistence gates.
- SIS accepts only loopback IP literal Worker URLs, validates Worker's reported listener/auth host/port contract, rejects redirects and arbitrary request-supplied paths/URLs, uses exact configured workspace IDs, and does not send conversation history/provider credentials. Workspaces, backends, and model routes come from Hydra only.
- An opt-in Worker API auth implementation is present in the checkout, but the running Worker is a legacy instance without that contract and no Worker/SIS API token is provisioned. The current process therefore remains unauthenticated, although it is confirmed loopback-only. The new SIS gate requires auth by default.
- Chat and Worker execution remain disabled; the local Worker route is pinned to GPT-6 Luna and its inference-verification assertion is true after the bounded check. No workspace is registered. A workspace attestation is required, but code does not create an OS sandbox. Worker `opencode` has no request-level read-only enforcement; it may modify files, run commands or access network resources subject to host configuration. Do not enable against the Stephanie working tree or sensitive data.
- Result/task content is bounded and credential-like data is rejected/redacted. No external side-effect approval flow is implemented; no commit/push/publication capability was independently granted by SIS.

## G. Tests

Commands:

```powershell
venv\Scripts\python.exe -m pytest tests/test_sis_worker_execution.py -q
venv\Scripts\python.exe -m pytest tests/test_sis_chat_execution.py tests/test_sis_security.py -q
venv\Scripts\python.exe -m pytest --ignore=tests/test_elm.py -q
bun run test:worker
bun run typecheck:worker
```

Results: STN03 focused tests **20 passed**; STN02 focused/security baseline **38 passed** (combined focused run: **58 passed**); the full Worker test suite **211 passed, 8 skipped**, with Worker typecheck passing. The broader Stephanie run excluding `tests/test_elm.py` reported **117 passed, 3 failed**, all three still require the absent external chapter fixture at `C:\Projects\new-books\...\07-chapter.md`. Full collection still aborts on the known `AxisDirection` import mismatch and `CalibratedThresholds` NameError in `tests/test_elm.py`.

The non-writing `venv\Scripts\python.exe scripts\stn01_smoke.py` command completed and explicitly reported that it made no database connection or writes. The earlier STN01 LoaderAgent/PlanTrace execution remains the available workflow evidence; this non-writing invocation does not claim a new pipeline run.

## H. Limitations and live gate

Not implemented: streaming, a persistent/continuable OpenCode session, verified native permission responses, reliable cancellation, durable Worker recovery, an OS sandbox, automatic model-route failover, or Work acceptance in Stephanie. Worker API bearer auth is now active and verified for the live dry-run path. `AWAITING_ACCEPTANCE` is never presented as verified success; SIS labels the exact no-invocation response `DRY_RUN_NO_AGENT_EXECUTION`. Timeouts and lost responses are not automatically resubmitted.

The basic Worker result does not expose a structured token-usage/cost field, and may not report an observed model identity. SIS records only the explicit requested route and observed provenance/duration; it never infers task tokens or cost from output. The separate GPT-6 Luna preflight did report usage/cost in the OpenCode session export, as recorded above.

Authenticated transport acceptance **PASSED**: one real SIS dry-run produced Worker request `wrq_8138873113779fbaa7e04a5afa55c58e`, linked to conversation `1666`, execution `sex_d17f254a7c2c4875bdf9054a356de8cd`, and PlanTrace `worker:sex_d17f254a7c2c4875bdf9054a356de8cd`. Fresh PostgreSQL readback matched all identities. No coding task or model inference was run. Live coding acceptance remains **BLOCKED** because no filesystem/process/network/credential sandbox is verified; Docker daemon is unavailable and its service is stopped/manual.

## I. Next milestone

Keep the authenticated Worker/SIS dry-run services active with `worker.enabled: false`. Next investigate Docker Desktop startup only with operator approval if elevation is needed; validate filesystem/process/network/secret/resource restrictions and OpenCode compatibility before enabling any real coding task. Only then consider native OpenCode session continuation/streaming. Durable Worker recovery and cancellation semantics remain gaps; Pi integration is not justified yet.
