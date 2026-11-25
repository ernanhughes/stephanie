# SC4 completion and conversational-routing report

## Independent verdicts

| Workstream | Verdict |
|---|---|
| SC4A functional integration | Implemented in source and exercised within isolated tests; not an operational release package. |
| SC4B isolated qualification | The bounded isolated workflow passed. Owner-review experiment `7cc37cb2e95dd58d098c3d80` remains reported complete; no formal post-result decision was recorded. Negative controls remain partial. |
| SC4C operational activation | **Blocked; OFF.** No deployment, migration, operational write, or restart was performed. |
| Free-first routing implementation | Deterministic SIS routing foundation implemented on the existing async `LLMService`/LiteLLM path. SIS chat stays disabled. |
| Actual OpenCode Free compatibility | Both selected free routes each passed one synthetic CLI invocation. OpenCode documents direct application API support; Stephanie's API path is not live-qualified. No paid calls. |
| Conversational continuity | Existing ChatStore/SIS history, idempotency and context path are reused. Route changes preserve the supplied message history. Persisted live-turn/restart acceptance with OpenCode remains outstanding. |

## Changes

- Stephanie: added config-gated `FreeFirstChatRouter`, integrated it with SIS `LLMService.achat`, added two explicitly policy-eligible free routes in `config/sis.yaml`, added deterministic routing tests, and added the eight requested SC4/routing records. No new conversational database or model registry was introduced.
- Mission: no backend behavior change in this continuation; corrected one stale Improvement UI comment about the generic manual handoff. The existing dedicated SC4 integration accepts human-approved proposals directly through matching envelope registration and dispatch; no file transfer or automatic approval is introduced.
- AI, Worker, Praxis: no changes in this continuation. Their previously dirty work remains outside this delivery.

## Verification

- Stephanie: 13 focused routing and adjacent SIS tests passed in 8.85 seconds including pytest startup (slowest individual test 0.02s); SIS YAML parsed with `chat_runtime.enabled: false`; modified Python modules compiled. No live inference in these tests.
- Mission: prior focused SC4 API/envelope tests (12 passed), focused Improvement UI tests (4 passed) and frontend build (passed) remain recorded in the SC4 evidence. No Mission runtime behavior was changed here.
- SC4 long integration test: 1 passed in 61.23 seconds, per `SC4B-END-TO-END-EVIDENCE.md`; it used deterministic Worker fixtures and zero model calls.
- Real CLI isolation test: 1 passed in 22.55 seconds under process observation. Isolated artifacts and SQLite were in the pytest temp root; the test does not verify operational DB/ledger preservation or backup/restore. Detailed command, paths, observed process IDs and watcher limitations are in `SC4B-END-TO-END-EVIDENCE.md`.
- Long tests intentionally run: the exact SC4 integrated acceptance and the exact real-CLI isolation test. No other long suites were run.

## Operational and routing boundaries

On 2026-11-25, `:8790/api/health` returned `ok`, version `0.1.0`, database `ok`, schema 54. Read-only improvement APIs were present; their investigation integration reports unavailable without a configured isolated run root. The exposed deterministic record is a different artifact and retains formal acceptance `UNKNOWN`. The separate `:8791` review process remains available. No production dispatch or status mutation was attempted.

SC4 outstanding items: missing Worker service control, direct stale-version/revocation tests, multi-process dispatch, forced experience-retention failure, clean source/package provenance, exact database/configuration identity, usable backup and restore rehearsal, rollback and compatibility qualification, and a final independent review. Activation remains blocked.

Routing limitations: only ordinary conversation capability is configured. Capability escalation to coding, independent review, tool use and project-aware retrieval, owner UI pins, cross-process cooldowns, and live direct OpenCode API authentication remain unqualified. Data privacy class is a runtime configuration, not a per-turn choice; confidential source and protected evaluation material require local-only routing until that is added. No paid fallback is configured; only the selected free routes and the existing local Ollama fallback are eligible.

See `SC4-NEGATIVE-CONTROL-EVIDENCE.md`, `SC4C-ACTIVATION-PACKAGE.md`, `BACKUP-AND-ROLLBACK-EVIDENCE.md`, `INDEPENDENT-ACTIVATION-REVIEW.md`, `VERIFIED-OPERATIONAL-GUIDE.md`, `FREE-FIRST-ROUTING-IMPLEMENTATION.md`, `OPENCODE-FREE-QUALIFICATION.md`, and `CONVERSATIONAL-INTEGRATION-INPUTS.md`.

No commit or push was made. Do not activate chat, dispatch from `:8790`, promote a candidate, or commence SC5/SE3 under this report.
