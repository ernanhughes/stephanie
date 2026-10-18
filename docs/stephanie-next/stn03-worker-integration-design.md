# STN03 — SIS to Worker Integration Design

**Status:** Authenticated SIS → Worker dry-run transport acceptance passed and persisted through a fresh PostgreSQL connection. Ordinary chat and general coding execution remain disabled; real coding is blocked on verified isolation. No migration, commit or push was made.

## Ownership and request path

```text
Authenticated SIS conversation
  → explicit “Run with OpenCode” action
  → WorkerExecutionService (bounded thread pool, idempotent execution ID)
  → Worker POST /v1/work (worker.work-request.v1)
  → selected Worker executor
  → versioned worker.work-result.v1
  → ChatStore conversation metadata + user/assistant messages/turn
  → existing PlanTraceStore (external execution trace)
```

SIS owns the conversation, account authorization, workspace registration, Stephanie execution ID, state and conversation/evidence relationship. Worker owns request normalization, Worker request ID, executor invocation and Worker result semantics. The basic Worker `opencode` executor invokes the installed CLI; it is not an interactive OpenCode session.

SIS sends only the task, exact registered workspace root/project ID, explicit backend and model route, `acceptance: human`, and minimal correlation metadata (`source`, Stephanie `operation_id`, `stage_id`). It does not send ChatStore history, a Hydra tree, provider credentials, or the whole conversation. The operation ID is a Stephanie-generated UUID and is distinct from the Worker's returned `wrq_…` ID, the OpenCode `session_id`, and PlanTrace ID `worker:<Stephanie execution ID>`.

## Configuration and activation

`config/sis.yaml` adds a `worker` node. Checked-in values are intentionally:

```yaml
worker:
  enabled: false
  base_url: http://127.0.0.1:7317
  api_token_env: STEPHANIE_WORKER_API_TOKEN
  require_api_auth: true
  timeout_seconds: 600
  default_backend: opencode
  allowed_backends: [opencode]
  model: opencode-go/gpt-6-luna
  readiness:
    model_verified: true  # This local checkout passed a bounded direct OpenCode probe.
  max_concurrent: 2
  max_queued: 2
  max_records_per_conversation: 100
  workspaces: {}
```

This checkout now explicitly selects `opencode-go/gpt-6-luna`; one direct, tools-denied OpenCode CLI inference verified that exact route on this account. `readiness.model_verified: true` records that local observation. It does **not** turn on execution: `enabled` remains false, the Worker API token is absent, no workspace is registered, and this route's state must be re-verified on any other machine/account. In general an operator must verify an exact route through a real bounded inference before setting `model_verified: true`. The workspace flag is also an operator attestation—not a technical sandbox proof. It must only be set for a disposable, non-sensitive workspace whose filesystem, process execution, network egress, and secrets are isolated outside SIS/Worker. SIS never accepts a workspace path from chat or a request parameter.

Real coding execution also requires the existing `chat_runtime` feature and model/persistence gates, configured SIS authentication, a ready database, conversation ownership, CSRF/Origin checks, an explicitly allowed backend, and a registered externally verified workspace. Ordinary chat is never routed to Worker automatically. No Worker process is started by SIS.

## Authenticated transport dry-run

The SIS request contract has an explicit `dry_run` flag (defaulted on for the API/UI). A dry run requires a registered workspace marked `dry_run_only: true` or an externally verified workspace; it still requires the authenticated Worker API, explicit backend/model route, and SIS account/session. For dry-run readiness, SIS calls `/health/transport`, which reports listener/auth/backend registration without probing OpenCode, and does not call Worker model discovery. Worker core validates the selected executor registration, records the Work/job, and returns `AWAITING_ACCEPTANCE` with `dry run: OpenCode was not invoked` **before executor health or execution**. Thus the dry-run path does not spawn the OpenCode CLI, infer, or execute project tools. SIS maps only that exact no-output/no-session response to `DRY_RUN_NO_AGENT_EXECUTION`; it creates no assistant answer/ChatTurn and displays `DRY RUN — NO AGENT EXECUTION`. Any unexpected dry-run result fails the contract instead of being labelled safe.

The separate real-execution path remains disabled unless the workspace has `execution_boundary_verified: true`; a dry-run-only workspace can never authorize a real run. Worker `/health` is a separate executor probe and is not called for the dry-run action. The real route remains subject to the earlier filesystem/process/network/secret isolation requirement.

For operations that must be tested while ordinary chat/model gates remain off, `worker.dry_run_only: true` enables the separate authenticated `POST /chats/worker-dry-runs` action. It requires SIS authentication, a ready database, a matching authenticated Worker transport, and a registered dry-run-only workspace; it creates a new owner-linked conversation and external execution record without enabling chat. In this mode `worker.enabled` stays false, the ordinary execution route remains unavailable, and service-level checks reject every non-dry-run request. `config/sis_ops_local.yaml` is the local opt-in profile: `SIS_CONFIG_NAME=sis_ops_local` selects it while the checked-in `config/sis.yaml` remains production-disabled. The profile uses an empty temp fixture and is not a coding sandbox.

The Worker URL parser accepts only explicit `http` loopback IP literals (`127.0.0.1` or `::1`) with a port; redirects and ambient HTTP proxy settings are disabled. Worker health reports its listener address and whether API auth is enabled, which SIS validates. The Worker checkout now has opt-in bearer auth (`api_auth.token_env`): when enabled, `/v1/work`, job/result/context retrieval, provider-state reads and model-runtime endpoints require a bearer token. Worker CLI clients can use the same configured environment variable. It is disabled by default for existing consumers. `worker serve` refuses to reuse a listener whose bind/auth contract does not match its configuration. The currently running legacy Worker process has not been restarted or provisioned with a token and remains unauthenticated. Loopback and a bearer token do not isolate another process running as the same user; the execution sandbox remains a separate requirement.

To enable this boundary, an operator must place the same random token in the Worker token environment variable named by `api_auth.token_env` and SIS's `worker.api_token_env`. Keep the token in an approved secret store/process environment; never put its value in YAML, request metadata, reports or logs. Tokens must be at least 32 bytes. SIS defaults to requiring Worker API auth when execution is enabled; an old/no-auth Worker is rejected.

## Persistence, idempotency and status

No new ORM, table, or DDL is introduced. `ChatStore` atomically appends the task user message and stores bounded execution metadata under `chat_conversations.meta.sis_executions`; a separate metadata index maps the SIS submission idempotency key to that execution. A retry with the same key and identical request returns the existing execution; reusing it for a different request conflicts. The app never replays an uncertain Worker POST.

The external trace uses existing `PlanTraceStore` / `PlanTraceORM` with an external trace ID and no fabricated Goal or PipelineRun. It records submission and outcome steps, bounded summaries, Worker/OpenCode references, status and observed duration. The current basic `worker.work-result.v1` path does not expose a structured token-usage/cost field; SIS leaves those values unknown rather than deriving them from text. It also distinguishes the requested model route from a separately observed model identity. It does not persist token-by-token events or the complete agent transcript. A real Worker result with output is appended as the assistant message and ChatTurn; a failure does not create a fictitious assistant response.

When that assistant message is later included in ordinary chat context, `chat_context` adds a bounded attribution marker identifying it as external Worker output, with its execution/Worker IDs, requested model route and unverified status. This preserves the existing role-based ChatStore format without presenting the agent result as a verified Stephanie-authored answer or injecting a full transcript.

| Worker result | SIS status |
|---|---|
| Explicit dry run; Worker says OpenCode was not invoked, with no output/session | `DRY_RUN_NO_AGENT_EXECUTION` |
| `AWAITING_ACCEPTANCE` | `COMPLETED_UNVERIFIED` |
| `ACCEPTED` with a supplied acceptance ID | `ACCEPTED` |
| `ACCEPTED` without an acceptance ID | `COMPLETED_UNVERIFIED` |
| `REJECTED`, `BLOCKED`, `FAILED` | Same status |
| Lost/invalid/unknown outcome | `INTERRUPTED` / failure; no automatic resubmission |

The dispatcher keeps FastAPI handlers non-blocking and bounds concurrent plus queued work. SIS polls Worker job state by its correlation ID to retain the Worker identity before completion. If SIS restarts while the same Worker still retains that job, a later status read can reconcile it. Worker retains only a bounded in-memory job/result map; Worker restart/eviction can lose that result. SIS then marks a missing correlated job interrupted when it can reach Worker. This is not durable Worker recovery.

## Workspace safety boundary

The initial UI permission class is `inspect`, but that label is not itself enforcement. Worker’s basic `opencode` executor invokes a CLI with the permissions/configuration available to that OpenCode process. The inspected Worker request contract has no read-only workspace permission field, and a workspace path does not constrain subprocesses. A git worktree is not a security sandbox. Therefore the integration does **not** claim read-only execution or containment. Do not enable it for the real Stephanie tree or a sensitive workspace. A future verified external sandbox must restrict writes, process authority, network access, and visible secrets before the operator attestation can be truthful.

No commit, push, publication, remote mutation, streaming, native session continuation, or SIS cancel control is implemented. Worker’s async CLI process can be interrupted best-effort when its HTTP request is disconnected; process-tree termination is not guaranteed. SIS exposes no guaranteed cancellation action.

## Windows operation

From `C:\Projects\worker`, after an operator has reviewed its config and model route:

```powershell
bun run serve
bun run health
```

`serve` reuses an existing Worker only when health confirms a compatible version, loopback bind and matching API-auth mode; it refuses older/unverifiable listeners instead of starting a duplicate. Worker API auth remains off unless its operator config names a token environment variable. From `C:\Project\stephanie`, start SIS using its normal local entry point:

```powershell
venv\Scripts\python.exe -m sis
```

SIS startup does not probe or spawn Worker unless the explicit feature is used. Test commands are recorded in `stn03-worker-integration-results.md`.
