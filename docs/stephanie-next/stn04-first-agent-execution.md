# STN04 — First Real Stephanie Agent Execution

**Date:** 2026-10-17  
**Result:** PASS for the first controlled Docker/OpenCode/GPT-6 Luna task and authenticated SIS persistence.  
**Worker/SIS status:** `AWAITING_ACCEPTANCE` / `COMPLETED_UNVERIFIED`. A later authenticated SIS human review accepted only the `calculator.py` scope; no Worker acceptance ID was created.

## A. Runtime

- Docker Desktop **4.92.0**, context `desktop-linux`.
- Docker Engine **29.8.0**, Linux `amd64` backend (Docker Desktop/WSL2).
- `docker version`, `docker info`, `docker ps`, and disposable Linux-container execution succeeded.
- The pre-existing `devcontainer-db-1` PostgreSQL container was left running and untouched. No other existing container, host virtualization setting, or host network setting was changed.
- Worker **0.2.0** listened only on `127.0.0.1:7317`. `GET /v1/jobs` returned HTTP 200 with no bearer header; `/health/transport` reported `api_auth_required: false`.

## B. OpenCode and model

- The official Linux/amd64 image tag `ghcr.io/anomalyco/opencode:2.0.29` was verified and pinned to manifest digest `sha256:b386308c48be366cab92c98b5c764665f42f32dc9fc49866c9372fa59ecaefe5`.
- The small derived image includes only that OpenCode binary and Python for the calculator fixture. Its pinned Python base is `python@sha256:ba27df8b6499cc3756cff167690f3fcb5321faa1703b3a914df5f4fdadb32f05`. The locally built image ID used for execution was `sha256:38600fd029222f69011e600ca9dddfe1c216bc1e1f533136eacab6d7182a5caa`.
- Build recipe: `C:/Projects/worker/docker/opencode-python/Dockerfile`.
- Installed CLI reported **OpenCode v2.0.29** inside the container.
- A final-image tools-denied run returned exactly `OK` with model `opencode-go/gpt-6-luna`, provider HTTP 200, and OpenCode session `ses_eb5260a96ffepxfLsqdSF3UEQc`. Its configuration denied all tools and mounted no host project.
- The coding task also produced an OpenCode session and a confirmed HTTP 200 response from the explicitly selected OpenCode Go model. No fallback model was configured or observed.

## C. Worker implementation and routing

- Added the opt-in `opencode-docker` backend through Worker’s existing executor registry in `C:/Projects/worker/src/executors/opencode-docker.ts`.
- The executor validates the exact registered project root, pins the model to `opencode-go/gpt-6-luna`, starts OpenCode inside Docker using argument arrays, records container/session/provider/test identity, bounds output and duration, and removes per-task containers/network before returning success.
- Worker’s default executor remains `opencode`; the Docker executor is selected explicitly. A missing daemon, credential, image, workspace, or relay blocks the request. There is no host-execution fallback.
- The executor runs a configured independent verifier in a second, read-only, `--network none` container. For this fixture the only accepted verifier is `python3 -m unittest`.
- Runtime-only Worker configuration is at `%LOCALAPPDATA%\Temp\opencode\worker-ops.json`; it registers one disposable workspace and one explicit model. It is not a checked-in default.
- SIS’s ordinary `config/sis.yaml` remains Worker-disabled. The local, ignored `config/sis_stn04_local.yaml` profile enables only `opencode-docker` for `stn04-calculator`; ordinary chat remains disabled. The SIS request uses the existing authenticated execution endpoint used by the **Run with OpenCode** control.
- Worker has no API-auth middleware, token, or bearer requirement. SIS retains its existing login, session, ownership, same-origin, and CSRF checks.

## D. Isolation and credential handling

The coding container used:

- Non-root UID/GID `10001:10001`; no privileged mode; all Linux capabilities dropped; `no-new-privileges`.
- Read-only root; only the disposable fixture mounted read/write; generated OpenCode configuration mounted read-only; no Docker socket or unrelated host mount.
- A per-task Docker `--internal` network. Direct external access and direct connection attempts to the host Worker port failed. The container had no provider key in its environment; it received only a dummy relay credential.
- Limits: memory/swap **1 GiB**, CPU **1.5**, PID **128**, 10-minute execution deadline, bounded Docker logs/output, and temporary writable storage. `/sys/fs/cgroup` reported the configured memory, CPU, and PID limits; `/proc/1/status` showed zero effective capabilities and `NoNewPrivs: 1`.
- A per-task, unprivileged relay container with lower memory/CPU/PID limits. It receives the one targeted OpenCode Go credential and forwards only the exact OpenCode Go model-list paths and GPT-6 Luna Responses path to the hard-coded upstream `opencode.ai:443`; it rejects other paths, methods, and model IDs. Relay request bodies and secrets are not logged.
- Synthetic negative tests verified that the relay returns HTTP 403 for an unapproved path and wrong model, and that an internal-network client cannot directly reach the Internet. Synthetic host data was not a real credential and was deleted after the test.
- After each execution the Worker process removes its temporary `OPENCODE_API_KEY` environment value; relay and agent containers, the task network, and temporary config/cid files are removed. All four recorded coding-task containers checked after execution were absent; no per-task network or Worker temp directory remained.

**Network limitation:** The agent itself has no external route. To reach GPT-6 Luna, the relay is attached to Docker’s default bridge for outbound connectivity. Docker bridge networking does **not** provide a provider-only IP allowlist; the relay’s enforced application-level policy limits requests to the specific provider host/path/model. No host firewall, routing, or virtualization policy was changed. This is recorded as an application-level egress boundary, not a Docker network-layer allowlist.

## E. First coding task

**Fixture:** `C:/Projects/stephanie-test-workspace` (not either live repository), containing only `README.md`, `calculator.py`, and `test_calculator.py`. Initial defect: `add(a, b)` returned `a - b`; both tests failed before the task.

**Objective:** “Inspect the calculator project. Find the failing test, correct the implementation, run the test suite, and explain what changed.”

The agent changed only `calculator.py` to return `a + b`. It reported two tests passing. Worker then independently ran `python3 -m unittest` in a separate read-only, network-disabled container; it exited **0** and reported `Ran 2 tests … OK`. The final fixture contains no bytecode directory or extra file. The tests were also run from the host environment with `PYTHONDONTWRITEBYTECODE=1`; both passed.

### Worker-level result

- Worker request: `wrq_36e76042dca45cfe680630c95e748241`
- Executor/model: `opencode-docker` / `opencode-go/gpt-6-luna`
- OpenCode session: `ses_eb53409dbffep6ZOA9v9l8UWJy`
- OpenCode container: `27ab489fa2ab663f611b62052a30ee4e5db4fc5f8aa662d5cebab20b06b084f4`
- Model relay container: `97abb34c48bbe10f192cf9e92570180b16584c28b435239870e4f28ae0a856c9`
- Test result: `python3 -m unittest`, passed, exit 0.
- Worker status: `AWAITING_ACCEPTANCE` (not accepted).

The first relay startup attempt (`wrq_e8994aac651b8d4a15a992c8766df858`) correctly returned `BLOCKED` before OpenCode started because the relay readiness check raced container startup. A bounded readiness poll fixed that narrow issue; the failed attempt did not change the fixture.

## F. Authenticated SIS execution

The local SIS account authenticated through the existing `/login` flow. The task was submitted with the existing SIS session, same-origin/CSRF token, owned conversation, registered workspace, and `workspace_write` permission class restricted to the Docker backend. The normal SIS chat model remained disabled.

- Conversation: `1667`
- Originating user message: `83557`
- SIS execution: `sex_8991baa0c2724f35878292c83651588f`
- Worker request: `wrq_ab52b2dab127f5e3b532b563e898eea6`
- Backend/model: `opencode-docker` / `opencode-go/gpt-6-luna`
- Provider request/model: `gpt-6-luna`; relay observed HTTP **200** from the configured Responses endpoint.
- OpenCode session: `ses_eb52c75e6ffeNNZUogl7AwoRgm`
- OpenCode container: `19205cfc5dbd9092c384c9efb19c479f7d004328891ffe9eb402fda9bed4e318`
- Model relay container: `517404f9e493c15ec64c974e898ab0491832421adf4074395f74817da12538e7`
- Independently verified file: `calculator.py`
- Verification: `python3 -m unittest`, `passed`, exit `0`.
- SIS duration: **23,925 ms**.
- SIS status: `COMPLETED_UNVERIFIED`; acceptance ID: **none**.
- Assistant message: `83558`; PlanTrace: `worker:sex_8991baa0c2724f35878292c83651588f`.
- Subsequent SIS human review: `ACCEPT`, reviewer `sis`, review ID `stn05-human-review-stn04-accept-001`, accepted scope `calculator.py`. This is a SIS review only; it did not change Worker status or create a Worker acceptance ID.

The retained Worker result was idempotently re-applied after a narrow SIS persistence-mapping fix so the verifier command/status/exit code are present in the execution record and PlanTrace. This was an evidence update only; it did **not** run another model task. The authenticated conversation and evidence pages then displayed the Worker request, container/model identities, and passing independent verifier.

## G. Model accounting

The requested model and provider response are observed and recorded. OpenCode CLI/provider output did not report token usage or cost for these coding runs; the relay intentionally does not log response bodies. Cost is therefore **not reported and not estimated**. SIS recorded the execution duration above.

## H. Testing and regressions

- Docker disposable-container and network-policy checks: passed.
- OpenCode v2.0.29, tools-denied GPT-6 Luna inference in the final pinned image: passed (`OK`, provider HTTP 200).
- Worker: `bun run typecheck:worker` passed; `bun run test:worker` passed (**210 passed, 8 skipped**).
- SIS focused suites: `pytest tests/test_sis_worker_execution.py tests/test_sis_chat_execution.py tests/test_sis_security.py -q` (**58 passed**, one existing Starlette deprecation warning).
- Fixture: baseline two tests failed; final independent Docker verifier and host test run both passed (2 tests).
- Authenticated SIS conversation, execution result page, evidence page, and fresh-process PostgreSQL readback: passed.
- Fresh-process persistence was read through a new `psycopg2` connection and raw PostgreSQL queries for conversation metadata, messages, turns, PlanTrace metadata, and execution steps. A minimal fresh SQLAlchemy ORM-only read encountered the repository’s existing unresolved `TrainingStatsORM` mapper relationship; raw database readback succeeded and verified all required relationships.
- No full Stephanie suite was rerun in STN04. The previous baseline records 117 passing and 3 known missing-fixture failures.

## I. Acceptance gates

| Gate | Result | Evidence |
|---|---|---|
| DKR-01 | PASS | Docker Desktop 4.92.0 / Linux Engine 29.8.0 operational. |
| DKR-02 | PASS | Disposable Linux agent, relay, and verifier containers ran. |
| DKR-03 | PASS | Memory, CPU, PID, deadline, and output limits applied and checked. |
| DKR-04 | PASS | Non-root, read-only root, one writable fixture mount, no socket/caps/privilege escalation; synthetic host data inaccessible. |
| DKR-05 | PASS with limitation | Internal agent egress blocked; relay rejects unauthorized paths/models. Relay needs bridge egress; not a Docker IP-level allowlist. |
| OC-01 | PASS | OpenCode v2.0.29 executed inside the pinned Linux image. |
| OC-02 | PASS | Tools-denied final-image inference returned `OK`; coding task received HTTP 200 from GPT-6 Luna. |
| OC-03 | PASS | OpenCode tools changed the fixture; separate read-only verifier passed. |
| WRK-01 | PASS | Worker job recorded `opencode-docker` container/model relay events. |
| WRK-02 | PASS | Real Worker result returned `AWAITING_ACCEPTANCE`. |
| WRK-03 | PASS | Explicit backend only; unavailable runtime blocks; no host fallback. |
| SIS-01 | PASS | Existing SIS login/session/CSRF and owned-conversation execution endpoint submitted the task. |
| SIS-02 | PASS | Authenticated conversation displays `COMPLETED_UNVERIFIED`. |
| SIS-03 | PASS | Assistant result and evidence page display the execution result. |
| EVD-01 | PASS | Conversation, message, SIS execution, Worker request, OpenCode session, and container IDs persisted. |
| EVD-02 | PASS | PlanTrace contains submission/result steps and verifier outcome. |
| EVD-03 | PASS | New Python process and new PostgreSQL connection verified linked rows/metadata. |
| CODE-01 | PASS | `calculator.py` changed from subtraction to addition. |
| CODE-02 | PASS | Both tests passed independently in a read-only Docker verifier and host test run. |
| CODE-03 | PASS | Final disposable workspace contains only the three fixture files; only `calculator.py` is a task artifact. No live repo was mounted. |
| REG-01 | PASS | Focused SIS/Worker suites and Worker typecheck passed; broader suite not rerun. |

## J. Remaining limitations and next stage

- Egress is constrained by the relay’s code-level host/path/model policy, not by a Docker provider-IP allowlist. A stronger host/network-layer allowlist requires separate authorization and configuration.
- The local Worker intentionally clears its temporary Go credential after the one execution. Its Docker executor consequently reports unavailable until an operator explicitly supplies the credential for another run. The Worker API remains unauthenticated and loopback-only.
- Provider usage/cost is unavailable from the recorded CLI/relay events. No estimate is claimed.
- The standard minimal fresh ORM import still encounters the pre-existing unresolved `TrainingStatsORM` mapper relationship; the required fresh database readback succeeded using direct PostgreSQL queries.
- No commits or pushes were made. Existing unrelated working-tree changes were preserved.

**Recommendation:** STN04’s execution path and durable evidence passed technical acceptance; the operator recorded a scoped SIS human review. STN05 has captured one deterministic CaseBook candidate from that accepted execution, but the lesson itself remains `PENDING_REVIEW` and no MemCube is active. Do not treat this one task as evidence of general coding reliability or autonomous self-improvement.
