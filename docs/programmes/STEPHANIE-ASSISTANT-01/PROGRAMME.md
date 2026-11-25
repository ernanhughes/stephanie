# STEPHANIE-ASSISTANT-01 — Persistent Conversational Intelligence

## Objective

Make the existing SIS chat usable as a persistent, project-aware Stephanie conversation. Keep ChatStore as the transcript authority, use Stephanie's existing `LLMService`/LiteLLM chat interface, and query Mission Control only through its read-only HTTP API.

## Implemented scope

- Isolated PostgreSQL database: `stephanie_assistant_01`.
- Existing SIS server-rendered chat pages and ChatStore lifecycle.
- Per-turn, assembled-context data classification and a two-part external approval gate (current form approval plus every included earlier turn carrying the same eligibility label).
- OpenCode Go DeepSeek V4.1 Flash as an opt-in route; local Ollama Mistral Small is the configured local route.
- Mission project list/detail projections via GET only, with source and observation time.
- Stable assistant identity, bounded same-conversation history, and model identity/routing metadata on each saved answer.

## Boundaries

No Mission writes, direct Mission database access, SC4C activation, Worker delegation, autonomous task execution, metered pay-as-you-go route, or new chat database. OpenCode Free remains unqualified for direct third-party API access and was quota-limited in the CLI probe. The local fallback has not passed a response-quality qualification.

## Acceptance status

The isolated interactive runtime passed a genuine OpenCode Go application call, persisted two turns, survived a fresh-process read, and served the SIS chat and Mission picker routes. Privacy/routing, continuity, and read-only projection checks passed. The release is **conditionally accepted** because local-only answer quality and independent review are incomplete, and it cannot send private project context externally.

See `FINAL-REPORT.md` for final review and current commit state.
