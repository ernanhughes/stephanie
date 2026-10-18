# STN03-OPS — Worker Activation and Dry-Run Results

**Outcome:** The authenticated SIS → Worker transport acceptance **PASSED** with a genuine dry-run. Worker and SIS are active on loopback with bearer/local-account authentication. Conversation `1666`, execution `sex_d17f254a7c2c4875bdf9054a356de8cd`, Worker request `wrq_8138873113779fbaa7e04a5afa55c58e`, and PlanTrace `worker:sex_d17f254a7c2c4875bdf9054a356de8cd` survived a fresh PostgreSQL process readback. Worker returned `AWAITING_ACCEPTANCE` with its explicit no-invocation message; SIS correctly displays **DRY RUN — NO AGENT EXECUTION**. No model inference, tool execution, or coding task occurred.

## 1. Worker process and reload decision

The pre-existing legacy Worker was safely replaced only after the operator authorized its exact process tree and a fresh all-completed-job check:

- Old process PID `54408` and launch wrappers `44740`/`56164` were stopped after confirming its eight jobs all had completion timestamps and it owned the only loopback listener. No active job was interrupted.
- Updated Worker `0.2.0` is now PID `34300`, bound to exactly one `127.0.0.1:7317` listener. `/health/transport` reports auth required, and `/health` reports `opencode` available. Worker production routing was not changed; the local non-secret config is `%LOCALAPPDATA%\Temp\opencode\worker-ops.json`.
- Bearer verification passed: unauthenticated `/v1/jobs` and `/v1/work/<id>` returned 401; the same authenticated routes returned 200 for jobs and 404 for an unknown result (auth passed, object absent). No token value was printed.
- `writer models health --start --json` had earlier reported `WORKER_READY` on the legacy payload and had not started/replaced the process. Source review confirms its supervisor only requires an `executors` array; Writer's outbound adapter also does not attach bearer auth. The Worker was therefore activated with the updated checkout launcher and the dedicated auth config, not via Writer's current `--start` path.

The separately authorized replacement completed; no duplicate listener remains.

## 2. Authentication provisioning

The checkout has opt-in Worker bearer auth. Worker config stores only an environment-variable name (`api_auth.token_env`); SIS config expects the same random token through `STEPHANIE_WORKER_API_TOKEN` and requires auth by default. The token must be at least 32 bytes. Worker protects Work submission, job/result/context access, provider-state reads and model-runtime API endpoints; health remains public and reports whether auth is required.

The operator ran the secure provisioning helper. SIS (listener PID `30708`) accepted the configured account password and invalid sessions were rejected. SIS authenticated to existing database `co`; its identity check passed. The Worker token was loaded by both services through environment variables, and Worker reports `api_auth_required: true`.

The approved current-user DPAPI provisioning helper is prepared at `%LOCALAPPDATA%\Temp\opencode\Provision-Stn03Secrets.ps1`. Run it from an interactive PowerShell under the same Windows account that will run SIS/Worker:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\Temp\opencode\Provision-Stn03Secrets.ps1"
```

It prompts without echo for a new local SIS account password and the existing PostgreSQL `co` password, generates the session secret and shared Worker token, and stores DPAPI ciphertext with current-user-only ACLs under `%LOCALAPPDATA%\Stephanie\Secrets\stn03-ops.json`. It displays no password/hash/token values. Do not paste secrets into chat. The matching non-secret Worker config is staged separately at `%LOCALAPPDATA%\Temp\opencode\worker-ops.json` and refers to `WORKER_API_TOKEN`.

Worker CLI `serve` and SIS both fail closed if their auth/listener settings do not match. Operator provisioning instructions are also in `docs/stephanie-next/stn02-security-design.md` and `C:\Projects\worker\docs\api-authentication.md`. Do not print or report plaintext secret values.

The secret bundle is stored as current-user DPAPI ciphertext under `%LOCALAPPDATA%\Stephanie\Secrets\stn03-ops.json`; the services receive decrypted process environment only. The bundle/config/logs were not printed or added to Git. SIS login returned 303, invalid-session access returned 401, and Worker unauthenticated protected endpoints returned 401.

## 3. Authenticated dry-run implementation

SIS accepts an explicit `dry_run` Work option and now has a distinct `/chats/worker-dry-runs` route for transport acceptance. The route requires SIS auth and a ready database, creates a new owner-linked conversation, and does not require the separate live-chat/model readiness assertions. `config/sis_ops_local.yaml` is a local opt-in profile selected with `SIS_CONFIG_NAME=sis_ops_local`; it has `worker.enabled: false`, `worker.dry_run_only: true`, and an empty temporary workspace marked `dry_run_only: true`. Checked-in `config/sis.yaml` remains unchanged and production-disabled. At the service layer, dry-run-only mode rejects every non-dry-run request.

The registered fixture was created at `C:\Users\ernan\AppData\Local\Temp\opencode\stn03-dry-run-fixture`; it contains no project source. SIS dry-run readiness calls Worker `/health/transport`, which reports listener/auth/backend registration without executor health or model discovery.

Worker core now handles `options.dry_run` after checking that the named backend is registered but **before executor health or execution**. It records the Worker job and returns `AWAITING_ACCEPTANCE` with `dry run: OpenCode was not invoked`; it does not run `opencode --version`, invoke an agent, infer, or execute project tools. A test verifies the executor’s health and execute methods are both untouched. SIS maps only that exact no-output/no-session response to `DRY_RUN_NO_AGENT_EXECUTION`, displays **DRY RUN — NO AGENT EXECUTION**, persists the task and PlanTrace, and does not create an assistant answer or ChatTurn. An unexpected response fails closed instead of being labeled as a dry-run success.

Live request used the specified objective, backend `opencode`, model `opencode-go/gpt-6-luna`, and the registered empty temp fixture. SIS returned terminal `DRY_RUN_NO_AGENT_EXECUTION` with `opencode_session_id: null` and no output. Worker independently returned `AWAITING_ACCEPTANCE`, `dry run: OpenCode was not invoked`, no output/session, and events `JOB_ACCEPTED`, `DRY_RUN_REQUESTED`, `DRY_RUN_NO_EXECUTOR_INVOKED`, `STAGE_COMPLETED`, `JOB_COMPLETED`; there were no `JOB_STARTED`/`STAGE_STARTED` events. SIS detail and evidence views displayed the explicit dry-run label.

Ordinary chat is still disabled (`chat_configured: false`, `chat_enabled: false`); Worker general execution remains disabled (`worker.enabled: false`). The local ops profile only enables `dry_run_only` and cannot submit non-dry-run work. The dry-run made no OpenCode inference, agent session, project tool call, or source modification.

## 4. Live identity and persistence readback

| Identity/evidence | Observed value |
|---|---|
| SIS conversation | `1666` (owner-linked; `sis_dry_run_only: true`) |
| Stephanie execution | `sex_d17f254a7c2c4875bdf9054a356de8cd` |
| Worker request | `wrq_8138873113779fbaa7e04a5afa55c58e` |
| Worker response | `AWAITING_ACCEPTANCE`; message confirms OpenCode was not invoked |
| SIS status | `DRY_RUN_NO_AGENT_EXECUTION` |
| Backend / requested model | `opencode` / `opencode-go/gpt-6-luna` |
| PlanTrace | `worker:sex_d17f254a7c2c4875bdf9054a356de8cd` |
| Fresh-connection readback | All conversation, execution, Worker request, status and PlanTrace identities matched; two trace steps; zero ChatTurns |

The fresh readback ran in a separate Python process with a new MemoryTool/PostgreSQL connection. SIS was not restarted; the new process read persisted evidence independently.

## 5. GPT-6 Luna state

The exact route `opencode-go/gpt-6-luna` was confirmed by both OpenCode and Worker model discovery and passed one direct tools-denied OpenCode inference check. OpenCode reported `OK`, 770 input/5 output tokens, `$0.0000795`, 2,937 ms, session `ses_eb947dfaeffeV0UUrGLHxJLn61`, and assistant message `msg_146b8242c001r18jELsjoL8qbg`. `config/sis.yaml` now selects this model and marks the model-only verification true; `worker.enabled` remains false. Worker's independent production routing policy was not modified.

The separate free-only LongCat probe returned `unavailable`/HTTP 503 and remains a recorded failed observation. It was not repeated. The GPT-6 Luna inference check used a separate metered CLI path, not that free-only probe.

## 6. Sandbox investigation

- Windows Sandbox executable: absent.
- Docker CLI 29.8.0 is installed, but the default `npipe:////./pipe/docker_engine` daemon endpoint is unavailable. `com.docker.service` is installed but stopped/manual; no Docker Desktop/backend process is running.
- WSL lists `docker-desktop` and `Ubuntu-24.04`, both stopped. No filesystem/process/network restrictions were established or tested.
- I did not start Docker Desktop or its Windows service: the service may require elevation and host-level startup/network integration, which was not authorized. No installation, virtualization, or network settings were changed.

No restricted container, VM or workspace was created. A WSL distribution, temporary directory or Git worktree alone is not an execution sandbox. **Real coding-agent execution remains BLOCKED** until an operator establishes and verifies filesystem, process, network-egress, credential and resource restrictions.

## 7. Acceptance snapshot

| Gate | Status | Evidence |
|---|---|---|
| Updated Worker active and loopback-bound | **PASS** | Worker `0.2.0`, PID 34300, exactly one listener at `127.0.0.1:7317`, OpenCode executor reports available. |
| Legacy Worker safe replacement | **PASS** | Eight completed jobs confirmed; operator authorized exact replacement; old process tree replaced without duplicate listener. |
| Worker API auth implementation | **PASS (source/tests)** | Optional bearer auth is implemented and covered by Worker and SIS tests. |
| Worker API auth active with provisioned token | **PASS** | Auth-required health; protected unauthenticated requests 401, authenticated requests accepted. |
| SIS account/session + PostgreSQL `co` | **PASS** | Real login succeeded; invalid session rejected; fresh app startup reports auth/database ready. |
| GPT-6 Luna model-only inference | **PASS** | Direct tools-denied OpenCode inference, actual identity/usage/cost recorded above. |
| SIS → authenticated Worker dry-run, no executor call | **PASS (live)** | Real SIS request, actual Worker response/events, no executor invocation, explicit SIS display. |
| Fresh PostgreSQL ChatStore/PlanTrace readback | **PASS (live)** | Separate Python process matched all IDs/statuses; no ChatTurns. |
| Live coding task in a verified sandbox | **BLOCKED** | No verified OS/process/network isolation. |
| Docker daemon/sandbox | **BLOCKED** | Docker service stopped/manual, daemon unavailable; no privileged startup/config changes made. |

## 8. Tests

- STN03 focused Worker/SIS tests: **20 passed**.
- STN02 chat/security tests: **38 passed**; combined focused suite: **58 passed**.
- Worker suite: **211 passed, 8 skipped**; `bun run typecheck:worker` passed.
- Broader Stephanie suite excluding `tests/test_elm.py`: **117 passed, 3 failed**, all from the known missing external chapter fixture.
- Full collection remains blocked by the known `AxisDirection` / `CalibratedThresholds` errors.

No schema migration, Docker configuration change, source-control commit or push occurred. One new SIS conversation and its execution/PlanTrace evidence were intentionally written to PostgreSQL `co` for this acceptance. The Worker replacement and SIS launch were intentional and are active. The local DPAPI bundle/config and temp workspace are outside Git; `config/sis_ops_local.yaml` is locally excluded. `sis/__main__.py` now calls its defined `main()` so the documented `python -m sis` launcher actually starts the service.

## Next action

**Next action for real coding:** start Docker Desktop only with operator approval if it requires elevated service startup; then validate a disposable container's filesystem mounts, process authority, network egress, inherited credentials, resource limits, teardown, and OpenCode compatibility. Keep real coding execution disabled until those restrictions are demonstrated. The authenticated transport acceptance is complete; it must not be described as an accepted coding task.
