# STN03-LIVE — Readiness and Live Acceptance

**Outcome:** GPT-6 Luna model-only verification and authenticated SIS → Worker **dry-run transport acceptance passed**. No coding task, model inference, or project tool ran. Ordinary chat and general Worker execution remain disabled. A separate Worker zero-priced LongCat probe failed HTTP 503 and was not retried. Real coding remains blocked on verified sandbox isolation.

## A. Runtime

- Updated Worker `0.2.0` is PID `34300`, with exactly one listener at `127.0.0.1:7317`; `/health/transport` confirms loopback bind and API auth required. `/health` reports `opencode` available; CLI `v2.0.29`.
- The legacy PID `54408` was replaced after verifying its eight jobs all had completion timestamps and receiving explicit operator authorization. No job was interrupted and no duplicate listener remains.
- `writer models health --start --json` previously reported `WORKER_READY` but did not restart the legacy process. Writer's `WorkerSupervisor` only checks that `/health` contains an executors array, and its outbound provider adapter sends no bearer token. The current Writer command is therefore not a valid replacement/authentication mechanism for SIS Worker activation.
- Worker auth verified live: unauthenticated `/v1/jobs` and result reads returned 401; authenticated job reads succeeded. SIS is PID `30708`, bound to `127.0.0.1:8000`.

## B. Model discovery and probe

Commands used:

```powershell
bun src/cli/cli.ts models list --free
bun src/cli/cli.ts models list
bun src/cli/cli.ts models status --all
bun src/cli/cli.ts models probe --model opencode-go/longcat-2.5-preview-free --max-requests 1
bun src/cli/cli.ts models status --all --model opencode-go/longcat-2.5-preview-free
opencode models
```

Discovery returned 14 models with explicit zero pricing. The named candidate `opencode-go/longcat-2.5-preview-free` was listed as free and `NOT_TESTED`; this is current discovery, distinct from the historical October 14 success. The one bounded probe actually ran through Worker’s tools-denied temporary probe environment:

- Invocation budget: one
- Result: `FAILED`, `unavailable`, HTTP 503
- Duration: 2.811 seconds
- Authentication, model availability and quota: `UNKNOWN`
- Worker local retry state: `BLOCKED` until `2026-10-16T21:25:31.852Z`

I did not refresh or retry during the cooldown or probe another free route. The configured DeepSeek route was not used. Worker CLI discovery (without `--free`) also reported `opencode-go/gpt-6-luna` as metered (`free: false`), so I checked it with a separate one-call OpenCode CLI inference—not the zero-priced-only Worker probe:

```powershell
opencode run --standalone --format json --model opencode-go/gpt-6-luna --title 'STN03-LIVE bounded metered probe' 'Reply exactly OK.'
```

The call ran from a unique temporary project with an OpenCode config `permission: "deny"`, explicit model and autoupdate disabled. `opencode debug config` confirmed the effective deny-all tool policy. Sanitized session export recorded:

| Observation | Value |
|---|---|
| Requested/observed provider/model | `opencode-go/gpt-6-luna`, variant `default` |
| Outcome / finish | `succeeded` / `stop` |
| Reply | `OK` |
| OpenCode session | `ses_eb947dfaeffeV0UUrGLHxJLn61` |
| Assistant message | `msg_146b8242c001r18jELsjoL8qbg` |
| Usage | 770 input, 5 output, 0 reasoning, 0 cached read/write tokens |
| Reported cost | `$0.0000795` |
| Duration | 2,937 ms |

The exported session contained user/assistant/idle messages and no tool call. This successful direct probe is not a Worker Work request, SIS conversation or PlanTrace.

## C. Security and isolation

- SIS account, session secret, shared Worker token, and existing PostgreSQL `co` password were provisioned through current-user DPAPI; plaintext values were not printed or committed. SIS login returned 303, invalid sessions returned 401, database identity/readiness passed.
- Checked-in `config/sis.yaml` remains disabled. SIS was launched with `SIS_CONFIG_NAME=sis_ops_local`; this local profile has `worker.enabled: false`, `worker.dry_run_only: true`, and one empty temp workspace. Ordinary chat remains disabled.
- Windows Sandbox is absent. Docker CLI 29.8.0 is installed but cannot connect to its daemon endpoint; `com.docker.service` is stopped/manual. Docker Desktop and Ubuntu WSL distributions are stopped. No service startup/elevation, virtualization, networking, or installation changes were made; no sandbox boundary has been verified.
- The basic Worker `opencode` executor has no per-request read-only guarantee. The operator workspace attestation remains unset, and Worker/SIS execution gates remain off. No external side effect or workspace mutation was attempted.

## D. Actual SIS execution

One real authenticated dry-run was submitted through SIS → adapter → authenticated Worker API. The objective was `Verify Stephanie-to-Worker connectivity without executing an agent.` The run used backend `opencode`, requested model `opencode-go/gpt-6-luna`, and the registered empty fixture. Worker result was `AWAITING_ACCEPTANCE` with `dry run: OpenCode was not invoked`; SIS correctly maps this to `DRY_RUN_NO_AGENT_EXECUTION`. No OpenCode session, output, inference, or tool execution exists.

| Identity | Value |
|---|---|
| SIS conversation | `1666` |
| Stephanie execution | `sex_d17f254a7c2c4875bdf9054a356de8cd` |
| Worker request | `wrq_8138873113779fbaa7e04a5afa55c58e` |
| PlanTrace | `worker:sex_d17f254a7c2c4875bdf9054a356de8cd` |
| SIS result label | `DRY RUN — NO AGENT EXECUTION` |

The rendered conversation detail and evidence pages display that label and the Worker identity.

## E. Persistence

After the request, a separate Python process opened a new PostgreSQL connection and read the conversation, execution metadata and PlanTrace. IDs/status/Worker request matched; PlanTrace has two steps; conversation has zero ChatTurns; no schema migration was run. SIS itself was not restarted; live persistence across a fresh database connection is verified, while SIS process-restart verification was unnecessary and not performed.

## F. Tests and regression results

- STN03/SIS focused integration tests: **20 passed**.
- STN02 chat/security focused tests: **38 passed**; combined focused set: **58 passed**.
- Worker suite: **211 passed, 8 skipped**; `bun run typecheck:worker` passed.
- Broader Stephanie suite excluding `tests/test_elm.py`: **117 passed, 3 failed**, all because `C:\Projects\new-books\content\books\dspy-from-first-principles\07-chapter.md` is absent.
- Full collection still aborts on the known `AxisDirection` import mismatch and `CalibratedThresholds` error in `tests/test_elm.py`.
- Hydra composes with `chat_runtime.enabled: false`, `worker.enabled: false`, explicit model `opencode-go/gpt-6-luna` and `worker.readiness.model_verified: true` following the direct bounded probe. The non-writing `scripts/stn01_smoke.py` invocation made no database connection or writes.

## G. Acceptance matrix

| Gate | Result | Evidence |
|---|---|---|
| LIVE-A1 Authenticated Worker HTTP service | **PASS** | Worker 0.2.0 PID 34300; loopback/auth fields verified; unauthorized protected routes return 401. |
| LIVE-A2 OpenCode executor health | **PASS** | Worker HTTP health reports `opencode` available; CLI v2.0.29. |
| LIVE-B1 Exact zero-priced model discovered | **PASS** | Current Worker catalog lists `opencode-go/longcat-2.5-preview-free` with `free: true`. |
| LIVE-B2 Bounded inference probe succeeds | **PASS** | Direct tools-denied OpenCode call returned `OK`; actual model identity, usage and reported cost were exported. |
| LIVE-C1 Dry-run fixture | **PASS** | Empty temp directory registered only in the local dry-run-only profile; not a coding sandbox. |
| LIVE-C2 Isolation mechanism verified | **BLOCKED** | Windows Sandbox absent; Docker daemon unavailable; no other process/filesystem/network boundary verified. |
| LIVE-D1 SIS account/database ready | **PASS** | Authenticated login and `co` DB identity verified. |
| LIVE-D2 Dry-run-only runtime profile | **PASS** | `worker.enabled=false`, `dry_run_only=true`, ordinary chat disabled. |
| LIVE-E1 Worker access boundary enforced | **PASS** | Shared bearer credential required for protected Worker API. |
| LIVE-F1 Authenticated SIS → Worker dry-run | **PASS (transport only)** | Real request/response identities recorded above. |
| LIVE-F2 Coding task executed | **BLOCKED by design** | No coding task was submitted; sandbox absent. |
| LIVE-F3 Model produced task output | **NOT CLAIMED** | Dry-run does not infer; GPT-6 Luna probe remains separate. |
| LIVE-F4 Result rendered in SIS | **PASS (dry-run evidence)** | Detail/evidence pages show explicit no-agent label. |
| LIVE-G1 Live conversation linkage persisted | **PASS (fresh DB connection)** | Conversation 1666 links to execution and Worker request. |
| LIVE-G2 Live PlanTrace read through fresh connection | **PASS** | Trace and two steps read from fresh PostgreSQL process. |
| LIVE-G3 SIS process restart | **NOT RUN** | Fresh DB process readback passed; SIS restart was not needed. |
| LIVE-H1 No new regression failures | **PASS** | Focused and Worker tests pass; broader failures match missing external fixture. |
| LIVE-H2 No unintended external effects | **PASS** | Only the requested bounded model probe ran; no agent task, workspace mutation, credential provisioning or Worker restart. |

## H. Next step

Do not enable real coding execution yet. The transport deadlock is resolved. Next, with authorization for any required host service startup, determine whether Docker Desktop can start its stopped daemon without changing virtualization or networking configuration; then verify disposable-container filesystem, process, egress, credential and resource restrictions plus OpenCode compatibility. Only after those boundaries pass should a tool-enabled GPT-6 Luna coding task be submitted.
