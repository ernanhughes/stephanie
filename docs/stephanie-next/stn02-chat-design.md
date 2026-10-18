# STN02 — SIS Chat Design

**Status:** implemented; deterministic and real local-model acceptance completed. The checked-in feature flag remains off.  
**Scope:** authenticated, single-user local SIS conversation using the existing ChatStore, LLMService, MemoryTool and Hydra configuration. No Pi/OpenCode/tools, new store or schema.

## Current path

```text
Browser
  → SISAuthMiddleware (Host, session, Origin and CSRF)
  → /chats route
  → ConversationExecutionService
  → ChatStore (configured MemoryTool sessionmaker)
  → configured ServiceContainer llm
  → existing async LLMService.achat / LiteLLM
  → ChatStore atomic assistant-message + ChatTurn commit
```

Normal conversation reads and writes continue to use `chat_conversations`, `chat_messages` and `chat_turns`. PostgreSQL identity remains Hydra `cfg.db`; the live acceptance used `co`. A basic chat turn does not create a PlanTrace or start a Supervisor workflow. That separation avoids implying ordinary dialogue is a workflow execution; STN03 can later add explicit conversation-to-run evidence links.

## Execution contract

`sis/services/chat_execution.py` now owns a turn rather than leaving orchestration in the route. It:

1. Validates the one configured local user's conversation ownership. New conversations store `sis_owner` in existing conversation metadata; owner-less legacy conversations remain readable to the sole configured account.
2. Serializes same-process turns for a conversation and uses a client `Idempotency-Key` to prevent duplicate user/assistant turns on retries.
3. Reads only the configured number of messages preceding the request and builds a role-filtered, character-bounded prompt. Only `user` and `assistant` history is passed; historical content is data, not privileged policy.
4. Resolves `llm` from the existing ServiceContainer and invokes async `LLMService.achat()` with Hydra's selected model and bounded timeout/output configuration. The synchronous `chat()` API remains available to historical consumers. No MemCube/CaseBook retrieval or tool access is added.
5. Persists the user prompt before inference. Provider failure marks that message `failed`; explicit cancellation marks it `cancelled`. Both return a retryable status and create no assistant message or completed turn.
6. Commits the genuine assistant message and its `ChatTurn` in one ChatStore transaction. Retry with the same key returns an already committed answer or retries the preserved failed prompt without duplicating it.

The new request/outcome identifiers and status live in existing message/conversation JSON metadata. Provider, requested/returned model, response ID, actual token usage if provided, latency, finish reason and cache status are recorded only when returned. No cost or missing field is fabricated.

The per-conversation lock and authentication session registry are process-local. Run a single SIS worker; multiple worker coordination is not implemented. Persisted message ordering remains serialized by ChatStore's PostgreSQL parent-row lock. Cancellation is best-effort at the LiteLLM coroutine/network layer; there is no durable job/restart recovery.

## Failure and UI behavior

- The composer disables repeat submission while a request is active, retains the draft on error, and retries with the same idempotency key. Editing the draft starts a new key.
- An authenticated cancel action cancels an active async provider request; cancellation persists a cancelled user-message status and no assistant/turn.
- A failed provider request preserves its user message and does not create a fake response.
- An unavailable LLM service and an unverified model are distinct from an inference failure.
- Model output is rendered through autoescaping in existing Jinja templates; it is never interpreted as raw HTML.
- The conversation list and detail page show the configured model and the separate readiness status. Chat controls render only when all gates pass.
- The LLMService omits an absent API key rather than passing an empty `Bearer` header. This fixed the configured local Ollama call.

## Readiness is deliberately explicit

`/healthz` exposes booleans, not credentials:

- `authentication_ready`: local account hash, session secret and host/origin allowlists are valid.
- `database_ready`: MemoryTool's live Hydra/ORM/direct identity check matches.
- `model_configured`: model name and API base exist.
- `model_verified`: operator assertion after an actual bounded inference test.
- `live_persistence_verified`: operator assertion after the authenticated end-to-end conversation/restart test.
- `chat_configured`: the existing `chat_runtime.enabled` flag.
- `chat_enabled`: all of the above gates are true.

All model/persistence readiness assertions and `chat_runtime.enabled` default to `false` in `config/sis.yaml`. STN02 tests and the acceptance runner set gates only in the isolated test process; no readiness values were promoted in checked-in config.

## Extension boundary for STN03

`ChatConversationORM.id` remains the conversation identity. Message request IDs and assistant provenance are sufficient to make a future explicit run reference attributable, but are not a generic execution ledger. Do not treat `sis_request_id` as a PlanTrace ID. STN03 should define a versioned conversation/run/evidence relationship, link actual workflow runs to PlanTrace, and keep ordinary chat turns lightweight.

## Persistence / migration decision

No schema change is required. Legacy IDs, metadata, messages, turn ratings and annotations are preserved. The integration fixture tests existing ChatStore retrieval and appends in isolated SQLite; the live acceptance adds seven clearly titled acceptance conversations to configured local database `co` (IDs `1659`–`1665`), including two real turns in conversation `1665`. No existing conversation was modified.

## Reproduction

```powershell
venv\Scripts\python.exe -m pytest tests/test_sis_chat_execution.py -q
venv\Scripts\python.exe -m pytest tests/test_sis_security.py -q
venv\Scripts\python.exe scripts\stn02_model_smoke.py --execute
venv\Scripts\python.exe scripts\stn02_live_acceptance.py --execute
```

The live acceptance script requires `STEPHANIE_DB_PASSWORD`. It generates an ephemeral test account/session secret in process memory, exercises the real authenticated SIS routes and configured model, then starts a separate Python process to read the ChatStore result back. It never prints or persists its test password/session secret. `--execute` creates one additional conversation; without it, both scripts perform no database/model operation.
