# Stephanie Next — STN00–STN06 Acceptance Matrix

“Pass” applies only to the named check; it does not imply broad production readiness. The initial rows preserve the STN00/STN01 baseline; current STN02 status is recorded below.

| Area / acceptance | Check and result | Status / evidence |
|---|---|---|
| Repository state | `git status --short --branch`, HEAD and branch | **Pass:** branch `elm`, revision `7584f47c9ef8acb5ee25f7b8bcd17c07d4999eee`; initially clean |
| No destructive changes | Review commands and git diff | **Pass:** no resets, drops, migrations or external effects; only local code/docs added |
| Core architecture recorded | Read `docs/design.md`, Supervisor, MemoryTool, container, Hydra and SIS paths | **Pass:** see `architecture-inventory.md` |
| Hydra preserved/composes | `venv\Scripts\python.exe` compose `config`, `sis`, `chat` | **Pass:** all three compose; SIS live-chat config resolves to disabled by default |
| SIS boot/import | `venv\Scripts\python.exe -c "import sis.main; ..."` | **Pass with caveat:** 22 routes imported; >45 sec first attempt, model components load, GLiNER2 absent/fallback; not a request or database-store smoke test |
| PostgreSQL configured connection | Read-only psycopg2 `SELECT 1` to `co`; schema names only | **Pass:** DB reachable; no rows/prompt contents read |
| ORM/store database alignment | `engine` read-only identity and table names vs Hydra config | **Fail/blocker:** ORM engine is `verity`, config is `co`; `verity` lacks `chat_conversations` |
| PlanTrace persistence | Source review only; no write/read test | **Not demonstrated:** requires ORM target resolution and schema validation; `plan_monitor` is disabled in SIS config |
| Existing Hydra workflow | No execution | **Not demonstrated / blocked:** Supervisor passes unsupported `logger` argument to `ServiceContainer` |
| Real configured model | Ollama TCP check at `localhost:11434` | **Unavailable:** no real model call made |
| Bus | NATS TCP check at `localhost:4222` | **Unavailable:** no NATS-backed task run made |
| Pi / OpenCode installation | CLI version/help | **Pass for discovery:** Pi `1.2.0`, RPC mode advertised; OpenCode `v2.0.29`, `serve`/`api` advertised. No task/session/server started |
| Baseline test collection | `python -m pytest --collect-only -q` | **Fail:** `tests/test_elm.py` import mismatch (`AxisDirection`) then `CalibratedThresholds` NameError and `SystemExit(1)` |
| Baseline narrowed suite | `python -m pytest --ignore=tests/test_elm.py -q` before slice | **Partial:** 59 passed, 3 failed; all failures need missing file at `C:\Projects\new-books\...\07-chapter.md` |
| Post-STN01 existing suite (excluding collection crash) | `venv\Scripts\python.exe -m pytest --ignore=tests/test_elm.py -q` | **Partial:** 81 passed, same 3 external-fixture failures; STN02 final result below |
| STN00/STN01 SIS chat contracts | `venv\Scripts\python.exe -m pytest tests/test_sis_chat_execution.py -q` | Historical: 22 passed before STN02; current combined tests below |
| SIS LLM service registration | Import app and `container.get("llm")`; health check | **Pass:** LLM is registered and reports healthy after fixing the missing container arg and stale NER import; no provider completion was attempted |
| Chat database restart | No live chat message was created | **Not run:** chat stays disabled; auth boundary and provider configuration remain prerequisites |
| Pi/OpenCode end-to-end | No runtime invocation | **Not run:** adapter milestones only |
| Self-improvement evidence | No controlled learning experiment | **Not run:** existing source systems are inventory only, no improvement claim |

## Test environment distinction

The repository `venv` is the STN01 reference interpreter (Python 3.13.15), has Hydra/OmegaConf/pytest, and reports `pip check` clean. The optional `test` dependency group declares pytest. PATH Python is also 3.13.15 but lacks Hydra/OmegaConf and reports a torch/compressed-tensors version conflict; use `venv\Scripts\python.exe` for repository commands.

STN01 reran the existing suite excluding `test_elm.py`; the three missing external-fixture failures remain unchanged from baseline.

## STN01 — Canonical database and verified Hydra run

| Gate | Evidence | Status |
|---|---|---|
| DB-01 Hydra and ORM target match | `MemoryTool.database_identity()` reports configured/ORM/direct `co` and matching identity | **PASS** |
| DB-02 MemoryTool and stores consistent | ChatStore and PlanTraceStore use MemoryTool's configured sessionmaker; existing chat data was read | **PASS** |
| DB-03 PlanTrace uses configured target | Run `10026` trace was read through a fresh MemoryTool connection from `co` | **PASS** |
| DB-04 Legacy chat readable | One existing conversation and six messages were retrieved; representative SQLite fixture also preserved IDs, order, provider, metadata, turns, stars and annotations | **PASS** |
| DB-05 No schema changes | No DDL, migration, `create_all`, drop, or data-copy ran. `verity` was untouched | **PASS** |
| RT-01 Supervisor construction | Constructor accepts logger and selected Hydra service profile | **PASS** |
| RT-02 Required services initialize | Loader and PlanTrace dependencies initialized; container shutdown completed | **PASS** |
| RT-03 Hydra workflow | Controlled one-stage variant of `config/code.yaml` ran existing `LoaderAgent` with a temporary JSONL fixture; one scorable loaded, no model/bus call | **PASS** |
| RT-04 Persisted PlanTrace | Run/trace `10026`: status `completed`, one execution step; retrieved after closing and reopening MemoryTool | **PASS** |
| SEC-01 Interactive chat disabled | Hydra composes `chat_runtime.enabled: false` | **PASS** |
| SEC-02 Local binding | `python -m sis` defaults to `127.0.0.1`; new chat execution remains disabled | **PASS for local development.** No remote auth boundary exists; do not bind remotely |
| TEST-01 Focused tests | `tests/test_sis_chat_execution.py`: 22 passed | **PASS** |
| TEST-02 Existing suite | 81 passed, 3 failed due the missing external chapter fixture; full collection still aborts in `test_elm.py` | **BLOCKED by pre-existing test/environment issues** |
| DOC-01 Operator docs | Installation and STN01 results documents updated | **PASS with README limitation** |
| Repeatable smoke helper | Default `scripts/stn01_smoke.py` does not connect/write; `--execute` produced run/trace `10026` | **PASS** |

### Execution notes

- The workflow used Hydra composition of `config/code.yaml`, the existing `critic_loader` agent, a temporary one-record JSONL fixture, `services: all`, disabled bus, and PlanTrace scoring/indexing disabled. No LLM was called.
- Run/trace `10026` is the final accepted execution. An earlier smoke trace `10021` is explicitly marked `incomplete` with a diagnostic note after STN01 fixed the trace-completion guard; its rows were preserved.
- One first compose-only attempt failed before pipeline-run insertion because a Hydra runtime path interpolation was accessed without a Hydra runtime context; the resulting unique goal record remains. No existing goal or conversation was edited.
- Ollama and NATS remained unavailable; this acceptance does not establish model-backed or NATS-backed execution.

## STN02 — Secure conversational assistant

| Gate | Evidence | Status |
|---|---|---|
| CHAT-01 Existing ChatStore canonical | Live SIS route uses `MemoryTool.chats`; configured database identity remains `co`; no new store | **PASS** |
| CHAT-02 Conversation creation | Authenticated, CSRF-protected `/chats` route creates through ChatStore and stores `sis_owner` in existing metadata | **PASS** |
| CHAT-03 User messages persist | Multi-turn acceptance conversation `1665`, user messages `83552`/`83554` at orders 0/2; fresh process readback | **PASS** |
| CHAT-04 Real assistant responses persist | `ollama/qwen3` responses `83553`/`83555` at orders 1/3; fresh process readback | **PASS** |
| CHAT-05 Turn/order correctness | ChatTurns `37314`/`37315` link the ordered user/assistant pairs; deterministic SQLite tests cover retries/cancellation | **PASS** |
| CHAT-06 Restart readback | `scripts/stn02_live_acceptance.py --readback-conversation-id 1665` opened a new Python process/MemoryTool and found all four messages + two turns | **PASS** |
| CHAT-07 Bounded context | Role filtering, message count/character cap tests; no MemCube/CaseBook injection | **PASS** |
| CHAT-08 Provider provenance | Actual `ollama`, `ollama/qwen3`, response usage/latency persisted from provider result | **PASS** |
| CHAT-09 Failure/retry/cancel coherence | Deterministic failure preserves one failed user message; retry is idempotent; explicit cancellation records cancelled state without assistant/turn | **PASS** |
| SEC-01 Local authentication | Single-account PBKDF2 verifier, opaque server-side sessions, expiry, rotation and login rate limiting; 8 auth/security test cases | **PASS for the implemented single-user local boundary** |
| SEC-02 Unauthorized mutations | Middleware rejects unauthenticated POSTs and protects all mounted routers | **PASS** |
| SEC-03 CSRF/cross-origin | Login and session CSRF, exact Origin/Host, untrusted Host, loopback/HTTPS tests | **PASS** |
| SEC-04 Sensitive SIS routes | All HTTP routes behind the same middleware; docs/OpenAPI disabled; only login, minimal health and static assets public | **PASS** |
| SEC-05 Default chat gate | Config remains disabled; auth-unconfigured app returns 503 for private routes; readiness reports separate auth/DB/model/persistence/feature states | **PASS; live enablement remains off** |
| MODEL-01 Provider configured/reachable | Ollama tags endpoint lists qwen3; real request through LLMService | **PASS** |
| MODEL-02 Multi-message inference | Two real authenticated SIS turns through async LLMService; follow-up received bounded prior user/assistant history; provider/model/usage/latency observed | **PASS** |
| MODEL-03 Real responses survive restart | Authenticated two-turn SIS request→ChatStore write; fresh process readback of conversation 1665 | **PASS** |
| COMP-01 Hydra remains usable | STN01 controlled LoaderAgent run/PlanTrace 10026 remains verified | **PASS** |
| COMP-02 Legacy chats readable | STN01 live sample + controlled legacy SQLite fixture tests; no old chat records altered | **PASS** |
| COMP-03 Regressions | 97 passed, same 3 missing-chapter fixture failures; full collection still fails in `test_elm.py` | **PASS for no-new-failure comparison; known suite limitations remain** |
| DOC-01 Results | STN02 design/security/results docs and setup guidance added | **PASS** |

### STN02 activation status

Implementation and real model/persistence acceptance passed. **The current app instance is not ready for ordinary users:** account/session environment variables are absent in this checkout, and checked-in `chat_runtime.enabled`, `model_verified` and `live_persistence_verified` remain false. The live acceptance used an ephemeral in-process test account and temporary in-memory readiness overrides only. Provision a local account, verify its operational config, then explicitly enable the flags before serving normal users.

## STN03 — Worker-backed coding execution

| Gate | Evidence | Status |
|---|---|---|
| WRK-01 Worker health integration | Updated Worker `0.2.0` reports loopback listener/auth contract; current listener PID 34300. | **PASS** |
| WRK-02 Existing Worker API reused | Adapter calls `/health`, `/v1/models/cli`, `/v1/work`, `/v1/work/:request_id`, `/v1/jobs`; no second OpenCode runtime in Stephanie. | **PASS** |
| WRK-03 Backend explicit | Hydra allowed/default backend and request payload must explicitly select an allowed route; no fallback. | **PASS** |
| WRK-04 Workspace authorization | SIS accepts only configured workspace IDs and resolves their configured directory; arbitrary path is not accepted. | **PASS (authorization)** |
| WRK-05 Request/result schema validation | Versioned Work request and `worker.work-result.v1` status/backend/request-ID validation tests, including unknown status. | **PASS** |
| WRK-06 Actual Worker identity retained | Fixture result/job correlation retains Worker `wrq_…` ID separately from SIS execution and OpenCode session IDs. | **PASS (fixture)** |
| CHAT-01 SIS action authentication/CSRF | Actual SIS route tested behind `SISAuthMiddleware`; missing/bad CSRF rejected. | **PASS (deterministic session)** |
| CHAT-02 Execution linked to conversation | Live dry-run conversation 1666 links to execution `sex_d17f254a7c2c4875bdf9054a356de8cd`; owner and Worker identity verified after fresh PostgreSQL connection. | **PASS (live)** |
| CHAT-03 Existing chat behavior preserved | STN02 focused/security regression suite rerun. | **PASS** |
| CHAT-04 Result shown in SIS | Actual authenticated SIS detail and evidence pages display `DRY RUN — NO AGENT EXECUTION` and Worker request ID. | **PASS (live dry-run)** |
| EVD-01 Execution evidence persisted | Conversation execution metadata and actual Worker request `wrq_8138873113779fbaa7e04a5afa55c58e` re-read from PostgreSQL using a new process. | **PASS (live)** |
| EVD-02 PlanTrace linkage verified | PlanTrace `worker:sex_d17f254a7c2c4875bdf9054a356de8cd` references matching conversation, execution and Worker request IDs; two steps read from fresh connection. | **PASS (live)** |
| EVD-03 Evidence survives new DB connection | Fresh Python process/MemoryTool readback; no ChatTurn or OpenCode session. | **PASS (live)** |
| SEC-01 Execution disabled by default | `config/sis.yaml` has `worker.enabled: false`, `require_api_auth: true`, no token/workspaces, and the explicit GPT-6 Luna route; its model verification is separate from execution enablement. | **PASS** |
| SEC-02 Unauthorized execution rejected | Actual invalid SIS session returned 401; Worker protected jobs/result routes returned 401 without token and accepted authorized access. | **PASS (live)** |
| SEC-03 Workspace boundaries verified | Workspace authorization is tested, but SIS/Worker do not supply an OS/process/network sandbox. Operator attestation alone is not proof. | **BLOCKED** |
| SEC-04 Unsafe external effects not granted | Basic OpenCode has no request-level read-only or independent external-action approval contract. Feature remains disabled. | **BLOCKED** |
| LIVE-01 Real Worker request completed | Actual authenticated `/v1/work` dry-run completed as Worker request `wrq_8138873113779fbaa7e04a5afa55c58e`; no coding task was executed. | **PASS (transport only)** |
| LIVE-02 Real OpenCode model response received | Separate tools-denied direct OpenCode probe returned `OK` from observed `opencode-go/gpt-6-luna`; the unrelated free LongCat Worker probe failed HTTP 503. No coding task response. | **PASS (model preflight only)** |
| LIVE-03 Conversation/evidence readback | Conversation `1666` and PlanTrace were independently read from PostgreSQL in a fresh process with matching IDs/status. | **PASS (live dry-run)** |
| REG-01 STN02 tests remain passing | `tests/test_sis_chat_execution.py tests/test_sis_security.py` rerun. | **PASS** |
| REG-02 Existing Hydra workflow remains usable | STN01 LoaderAgent/PlanTrace smoke previously passed; default no-op smoke rerun without `--execute` only confirms no-write behavior. | **PASS (prior workflow evidence)** |

### STN03 activation status

Authenticated transport acceptance **passed**: SIS submitted a genuine Worker dry-run and persisted/read back ChatStore and PlanTrace evidence. This is **not accepted for live coding use**: `worker.enabled` and STN02 chat gates remain disabled, the local profile permits dry-runs only, and no verified execution sandbox exists. See `stn03-operational-activation.md` for runtime IDs and evidence.

## STN03-OPS — Authenticated Worker dry-run preparation

| Gate | Evidence | Status |
|---|---|---|
| OPS-01 Existing Worker process/reload safety | Eight old jobs completed; exact listener tree replaced after operator authorization. Current Worker PID 34300 is the sole loopback listener. Writer `health --start` misclassified the legacy service and did not restart it. | **PASS** |
| OPS-02 Persistent SIS/Worker/DB auth provisioned | Current-user DPAPI credentials loaded; SIS login/invalid-session checks and Worker 401/authorized checks passed; database identity ready. | **PASS** |
| OPS-03 Dry-run cannot invoke OpenCode | Actual Worker events include `DRY_RUN_NO_EXECUTOR_INVOKED` and exclude job/stage start; result has no output or session. | **PASS (live)** |
| OPS-04 SIS dry-run request/result persistence | Dedicated `/chats/worker-dry-runs` route; chat and real execution remain disabled; actual SIS detail/evidence pages show the explicit label. | **PASS (live)** |
| OPS-05 Live authenticated SIS → Worker dry-run | Request `wrq_8138873113779fbaa7e04a5afa55c58e`; PostgreSQL fresh-process readback confirms conversation/execution/PlanTrace links. | **PASS (transport only)** |
| OPS-06 Verified agent sandbox | Docker client installed; daemon endpoint unavailable; Docker service stopped/manual; both WSL distros stopped. No boundary tested. | **BLOCKED** |

See `stn03-operational-activation.md` for the process review, dry-run contract and remaining operator actions.

## STN04 — First real Docker/OpenCode/GPT-6 Luna execution

| Gate | Evidence | Status |
|---|---|---|
| DKR-01/02 | Docker Desktop 4.92.0, Linux Engine 29.8.0; disposable Linux agent, relay, and verifier containers ran. | **PASS** |
| DKR-03/04 | Cgroup resource limits read in-container; non-root, read-only root, one fixture mount, dropped capabilities, no-new-privileges, no socket, synthetic unrelated host data inaccessible. | **PASS** |
| DKR-05 | Internal agent network blocks direct egress; GPT-6 Luna uses a per-task relay that rejects other provider paths/models. Relay attaches Docker bridge for outbound. | **PASS with documented limitation:** no Docker-level provider-IP allowlist. |
| OC-01/02 | Pinned OpenCode 2.0.29 container; final-image tools-denied probe returned `OK`; actual coding task confirmed explicit GPT-6 Luna Responses HTTP 200. | **PASS** |
| OC-03 / CODE-01/02 | `calculator.py` changed from subtraction to addition; independent read-only Docker `python3 -m unittest` exited 0; host run also passed 2/2. | **PASS** |
| WRK-01/02/03 | Worker dispatched to `opencode-docker`, returned `AWAITING_ACCEPTANCE`, and did not fall back to host OpenCode. Worker remains unauthenticated and loopback-only. | **PASS** |
| SIS-01/02/03 | Authenticated SIS conversation `1667`, execution `sex_8991baa0c2724f35878292c83651588f`; result displayed as `COMPLETED_UNVERIFIED`. | **PASS** |
| EVD-01/02/03 | Worker request `wrq_ab52b2dab127f5e3b532b563e898eea6`, OpenCode/container/relay identities and PlanTrace persisted; new Python/psycopg2 process read back conversation, messages, turns, execution, and trace. | **PASS** |
| CODE-03 | Only the disposable workspace was mounted. Final fixture has only README, calculator and tests; no live Stephanie/Worker repository was mounted. | **PASS** |
| REG-01 | Worker: 210 passed, 8 skipped, typecheck passed. SIS focused: 58 passed. Broader suite not rerun. | **PASS for focused suites; broader NOT TESTED** |
| HUMAN-01 | SIS human review `stn05-human-review-stn04-accept-001` accepted only `calculator.py`; Worker status remains `AWAITING_ACCEPTANCE` and no Worker acceptance ID was created. | **PASS (scoped SIS review)** |

See `stn04-first-agent-execution.md` for full identities, environment limits, relay limitation, and usage-accounting caveat.

## STN05 — Verified experience capture and controlled reuse

| Gate | Evidence | Status |
|---|---|---|
| AUD-01/02 | Existing CaseBook, MemCube, PlanTrace, Experiment/Metric stores and learning agents inventoried; broad chat import and legacy MemCube client are not used for this gated flow. | **PASS** |
| ORM-01/02 | Fresh normal `MemoryTool` registration/read queries work for required stores. Partial ORM-only imports can still fail because the package does not register every relationship target; this did not block the normal store path. | **PASS for normal service path; partial-import limitation documented** |
| REV-01 | SIS evidence page exposes test output, artifact references, and disclosure that no raw before/after patch artifact was persisted. | **PASS** |
| REV-02/03 | Authenticated CSRF-protected execution review `stn05-human-review-stn04-accept-001` recorded the operator's ACCEPT for `calculator.py`; idempotent audit readback confirms no Worker acceptance ID. | **PASS (live scoped human review)** |
| EXP-01–06 / CASE-01 | After scoped human acceptance, deterministic extraction persisted candidate `exp_33fd63f133b44b21275681e5af0aabb7` in CaseBook `1744`/Case `21680`/Goal `2986`; fresh normal MemoryTool readback verified source/Scorable/PlanTrace links. | **PASS (live candidate; lesson review pending)** |
| MEM-01–03 | Separate candidate review, approved-source MemCube promotion, active-only retrieval, and withdraw/supersede/stale filtering pass isolated-store tests. Real candidate remains pending, with no MemCube row. | **PASS on synthetic fixtures; real promotion NOT TESTED** |
| SIS-01 | Review, candidate, and read-only retrieval preview are available through existing authenticated SIS routes. Live STN04 review was recorded; checked-in defaults remain disabled. | **PASS; lesson review pending** |
| RET-01–04 | Approved-only MemCube retrieval, deterministic relevance score, bounded untrusted-context preview, and explicit experience IDs in future Worker context are tested with synthetic approved records. Live preview returned zero while STN04 lesson remains pending. | **PASS on synthetic fixtures; no real experience retrieved** |
| HYD-01 | Experience service is registered through the existing ServiceContainer and controlled by Hydra `experience_learning` settings; no model-assisted extraction at boot. | **PASS** |
| TST-01/02 | STN05 + focused SIS/Worker tests: 68 passed; Worker typecheck passed. | **PASS** |
| SCI-01–03 | STN06 protocol is specified; no A/B inference, paid experiment, or improvement claim was made. | **PASS for design; experiment NOT RUN** |

See `stn05-learning-inventory.md`, `stn05-experience-design.md`, `stn05-implementation-results.md`, and `stn06-experiment-plan.md`.

## STN06-PREP — Frozen, reproducible experience-transfer experiment

| Gate | Evidence | Status |
|---|---|---|
| REC-01/02 | Fresh read-only `MemoryTool` readback (`scripts/stn06_reconcile.py`) resolved the conflicting STN05 reports: review `ACCEPT`, Worker acceptance null, candidate `APPROVED`, MemCube `WITHDRAWN`/`reuse_eligible=false`, retrieval `false`, transfer `untested`. No record was edited. | **PASS** |
| REC-03 | Residual inconsistency documented, not rewritten: source execution `experience_status` still reads `ACTIVE`; the service now keeps the label consistent for future withdrawals only. | **PASS (documented; history preserved)** |
| FREEZE-01 | Frozen the exact approved candidate `exp_33fd63f133b44b21275681e5af0aabb7:v1`, hash `d4d720…`, lesson, applicability, limitations and provenance; refuses if the MemCube is globally active; no MemCube mutation. | **PASS (live)** |
| FREEZE-02 | Experiment-scoped selection returns the frozen experience for Arm B only; Arm A receives no context. Ordinary `retrieve_for_task` returned zero for both pilot tasks. | **PASS (live)** |
| AUTH-01 | `authorize_scope` records an experiment-only authorization (`global_reactivation=false`); live persistence requires the operator's reviewer identity and was deliberately not fabricated. | **PASS for mechanism; live record pending operator** |
| FIX-01 | Eight held-out tasks (4 near-transfer, 2 complex, 2 unrelated) with disjoint workspace/sealed-reference/sealed-oracle file sets; manifest hash `1c0d167f…`. | **PASS** |
| FIX-02 | Every sealed oracle actually ran: defective workspace fails, reference solution passes — **8/8 verified**. | **PASS (live, no model)** |
| FIX-03 | Workspace and sealed oracle/reference path sets never intersect (leakage control asserted in tests). | **PASS** |
| SCHED-01 | Stable SHA-256 arm schedule verified identical across `PYTHONHASHSEED` values; frozen run order recorded per task. | **PASS** |
| SCHED-02 | Preregistration hash reproducible across runs (volatile timestamps excluded): pilot and full produced `08deb572…`. | **PASS (live)** |
| DISP-01 | Dry-run built both-arm dispatch bodies (`worker.work-request.v1`, `opencode-docker`, `opencode-go/gpt-6-luna`, `workspace_write`) within the prompt bound, with experiment/task/arm identity in `metadata`. | **PASS (no model call)** |
| SCORE-01 | Scoring enforces oracle exit code, expected scope and prohibited paths; distinguished simulated failure from success. | **PASS** |
| STORE-01 | Preregistration + authorization persist into the existing `ExperimentStore` (experiment, variants A/B) on isolated SQLite; no new database. | **PASS** |
| BUDGET-01 | Pilot 4 / full 16 bounded executions, 600 s cap, one model request each, no retries/substitution/escalation; `$2` target explicitly non-enforceable. | **PASS (limits); monetary metering NOT AVAILABLE** |
| TST-01 | `tests/test_stn06_preregistration.py` + SIS/STN05 regressions: **74 passed**; `py_compile` and `git diff --check` clean. | **PASS** |
| RUN-01 | No paid model call, no trial/metric row, no MemCube reactivation, no commit. | **NOT RUN (by design)** |

See `stn06-preflight-results.md` and `stn06-preflight.json`. The experiment is
ready for operator-authorized execution; it remains **NOT RUN** until the
experiment-scoped authorization record and metered-run approval exist.
