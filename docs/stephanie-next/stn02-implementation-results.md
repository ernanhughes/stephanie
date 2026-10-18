# STN02 — Secure Interactive SIS Chat Results

**Decision:** STN02 implementation, security tests, real inference and persistent conversation acceptance passed. **The checked-in chat switch remains off; this machine has no persistent SIS account configured, so normal SIS users still see a fail-closed setup page until an operator provisions credentials and explicitly enables the verified gates.**  
**Branch/base:** `elm` / `7584f47c9ef8acb5ee25f7b8bcd17c07d4999eee`; STN00/STN01 changes were preserved. All work remains local and uncommitted.

## A. Existing infrastructure reused

- SIS remains the FastAPI/Jinja interface; all existing mounted HTTP routes now pass through a single access boundary.
- Existing `ChatStore`, SQLAlchemy chat ORM, `MemoryTool` sessionmaker and Hydra-configured PostgreSQL `co` remain canonical. No chat database/table or ORM model was added.
- `ServiceContainer.get("llm")` resolves the existing `LLMService`; synchronous `chat()` remains compatible and new async `achat()` delegates to LiteLLM for cancellable SIS requests. No alternate provider client or chat framework was introduced.
- A narrow `ConversationExecutionService` coordinates one turn outside the route. Plain chat does not start Supervisor or create a PlanTrace. Workflow-to-conversation evidence linking remains STN03.
- Context is bounded by configured message count and character budget, includes only user/assistant roles, and includes no automatic MemCube/CaseBook retrieval or tool execution.

## B. Implemented access boundary

- New `sis/security.py` implements a single configured local account, PBKDF2-SHA256 password verification, a random revocable server-side session ID, per-session CSRF token, expiry, logout revocation and login rate limiting.
- `SISAuthMiddleware` protects every mounted SIS HTTP route and websocket upgrade; only `/login`, minimal `/healthz` and `/static/*` are public. FastAPI docs/OpenAPI are disabled. The unauthenticated application fails closed with 503 when auth secrets are absent.
- Mutations require a same-origin Origin/Referer and CSRF token. Host is allowlisted and the Origin must exactly match the request origin. Remote HTTP is rejected; remote hosts must be explicitly configured and use HTTPS.
- Session cookies are host-only, HttpOnly, SameSite=Strict, eight-hour expiry, and Secure on HTTPS or when `SIS_COOKIE_SECURE=true`. A login form never sends credentials in the URL.
- New SIS conversations store the configured sole username in existing conversation metadata. Existing owner-less legacy conversations remain accessible to that single account. Multi-user account/role sharing is explicitly out of scope.
- All post-login pages are `no-store`; responses include `nosniff`, same-origin referrer and frame headers; HTTPS adds HSTS. Jinja continues to autoescape model text.
- `SISAuthSettings.ready`, database identity, model configured, model verified, persistence verified and feature configured are distinct states in `/healthz`. No secrets or endpoint credentials are returned.

Auth setup (PowerShell, per process; do not commit values):

```powershell
$env:STEPHANIE_DB_PASSWORD = "<database password>"
$env:SIS_AUTH_USERNAME = "stephanie"
$env:SIS_AUTH_PASSWORD_HASH = (& venv\Scripts\python.exe -m sis.security --hash-password)
$env:SIS_SESSION_SECRET = (& venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))")
python -m sis
```

The hash utility prompts twice without echoing the password. Defaults bind to `127.0.0.1:8000` and allow only loopback Host/Origin values. Behind HTTPS, set `SIS_COOKIE_SECURE=true` and configure the trusted proxy/allowed Host and Origin. SIS uses process-local sessions and per-conversation locks; run one worker.

## C. Conversation behavior

- Create, open and continue existing conversations through the existing chat templates.
- The UI displays the configured model and readiness reason; the composer appears only when every explicit gate passes.
- Same-origin fetches/forms receive CSRF tokens; the UI disables duplicate submits, retains the draft on error, retries with the same idempotency key, and can cancel an active async request.
- `ChatStore` appends a request-keyed user message once. Provider failures leave that message marked failed and create no assistant message/turn. Cancellation marks it cancelled; a retry with the same key reuses it.
- An authenticated per-request cancel route cancels an active async LiteLLM coroutine and records cancellation without a response. Deterministic tests cover the cancellation state; provider-side cancellation is best-effort and there is no durable job/restart recovery.
- Successful completion atomically appends the real assistant message and related `ChatTurn`. A provider/database failure is never reported as a completed turn.
- Assistant metadata contains only available provider/model/response ID/usage/latency/finish/cached data. Missing usage and provider values remain unknown.
- Context excludes system/tool transcript rows, uses a bounded recent window, and treats prior text as untrusted conversation content.

## D. Real model and end-to-end evidence

Ollama’s read-only tags endpoint was reachable and included `qwen3:latest`. A first LiteLLM attempt exposed an empty API-key bug (`Bearer `); `LLMService` now omits the `api_key` argument when no key is configured. An initial short-budget Qwen3 attempt returned no visible answer after reasoning consumed the limit; the configured prompt now requests non-thinking output and uses explicit timeout/output bounds. The final check uses the async LLMService path.

Final LLMService smoke command:

```powershell
venv\Scripts\python.exe scripts\stn02_model_smoke.py --execute
```

Observed real provider result: provider `ollama`, configured model `ollama/qwen3`, non-empty `OK` response, response ID recorded, 38 prompt / 116 completion / 154 total tokens, about 5.5 seconds. The smoke wrote no conversation or prompt-history row.

The authenticated SIS route was then exercised with TestClient and a process-ephemeral local account. The runner changed only the in-memory feature/readiness settings for that isolated test, performed real login + CSRF-protected conversation creation + two real model sends (the follow-up included bounded prior history), closed the app lifespan, then queried ChatStore from a **separate Python process**.

Accepted record:

| Evidence | Result |
|---|---|
| Configured/ORM/direct DB | `co`, identity matched |
| Conversation ID | `1665` (title prefix `STN02 real model acceptance`) |
| Message IDs/order | User `83552` at 0; assistant `83553` at 1; follow-up user `83554` at 2; assistant `83555` at 3 |
| ChatTurn IDs | `37314`, `37315` linking both ordered pairs |
| Model provenance | provider `ollama`, model `ollama/qwen3` |
| Follow-up usage | 94 prompt / 95 completion / 189 total tokens (includes bounded prior turn) |
| Follow-up latency | ~2.5 seconds |
| Fresh-process readback | conversation, ordered messages, completed assistant metadata and turn all found in `co` |

Six earlier successful acceptance conversations (`1659`–`1664`) were also created during incremental verification; all seven are test-labeled acceptance records. No historical conversation was changed.

## E. Files changed

- Security/auth: `sis/security.py`, `sis/routes/auth.py`, `sis/main.py`, `sis/templates/login.html`, `sis/templates/base.html`, `sis/template_utils.py`, `sis/__main__.py`.
- Chat execution/UI: `sis/services/chat_context.py`, `sis/services/chat_execution.py`, `sis/routes/chats.py`, `sis/templates/chats/{list,detail}.html`.
- Persistence/provider: `stephanie/memory/chat_store.py`, `stephanie/services/llm_service.py`, `config/sis.yaml`.
- Reproducible checks: `scripts/stn02_model_smoke.py`, `scripts/stn02_live_acceptance.py`, `tests/test_sis_security.py`, extended `tests/test_sis_chat_execution.py`.
- Documentation: `stn02-chat-design.md`, `stn02-security-design.md`, this report, and updated acceptance/milestone/architecture/installation docs.

## F. Database and data impact

- No schema/model migration, `create_all`, DDL, table/index change or data-copy operation was performed.
- Existing conversations/messages/turn IDs and legacy annotations were not rewritten.
- Seven test-labeled conversations, sixteen messages and eight turns were intentionally persisted in configured local `co` during incremental real-inference/readback acceptance. No PlanTrace was created for plain chat.
- STN01’s `verity` database remains untouched.

## G. Verification

| Check | Result |
|---|---|
| `venv\Scripts\python.exe -m pytest tests/test_sis_chat_execution.py tests/test_sis_security.py -q` | **38 passed** |
| `venv\Scripts\python.exe -m pytest --ignore=tests/test_elm.py -q` | **97 passed, 3 failed** on the same absent external chapter fixture as baseline |
| Full collection | Still blocked in `tests/test_elm.py` by the known `AxisDirection` / `CalibratedThresholds` errors |
| `venv\Scripts\python.exe -m pip check` | No broken requirements |
| `compileall` changed Python modules | Passed |
| `git diff --check` | Passed; Git emitted only LF→CRLF notices |
| Default feature configuration | `chat_runtime.enabled=false`, `model_verified=false`, `live_persistence_verified=false` |
| Default app with no auth env | `/healthz` reports auth not ready and chat disabled; private routes return 503; login shows setup state |

Exact broader test command:

```powershell
venv\Scripts\python.exe -m pytest --ignore=tests/test_elm.py -q
```

The original STN01 narrowed baseline was 81 passes plus the same three fixture failures; STN02 adds no new failures. Ollama was reachable and used for a bounded real call. NATS was not needed for the async request/response chat path.

## H. Remaining gates and next milestone

The implementation and real local model/persistence acceptance pass. The checked-in SIS config intentionally remains disabled, and the current environment has no persistent `SIS_AUTH_USERNAME`, `SIS_AUTH_PASSWORD_HASH` or `SIS_SESSION_SECRET`. Therefore **the deployed/local `python -m sis` instance is not currently live-chat-ready**: an operator must provision a durable local account/session secret and intentionally set the chat/readiness config after reviewing the run evidence. The test account is generated only in the acceptance process and is never printed or persisted.

STN03 conversation-to-execution contract/design work may begin independently with chat still disabled. Do not add Pi/OpenCode invocation here. Keep PlanTrace for actual workflows; attach future workflow/run references to conversations without treating every chat turn as a PlanTrace.
