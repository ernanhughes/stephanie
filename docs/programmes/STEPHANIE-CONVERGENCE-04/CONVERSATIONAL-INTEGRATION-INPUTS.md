# Conversational integration inputs

## Existing pieces

- SIS `ConversationExecutionService` owns request idempotency, per-conversation locking, owner checks, bounded history selection, and turn lifecycle.
- Existing `ChatStore` stores conversations, messages and turns. Prior messages are supplied to `LLMService.achat`; switching model routes does not change conversation identity or history.
- `build_chat_messages` applies a bounded system/history/context budget and treats prior/tool data as untrusted.
- Existing `LLMService` and LiteLLM provide sync/async chat calls and return observed model/usage/response provenance. The new free-first router uses the async path and caller-provided messages.
- Current SIS config remains disabled (`chat_runtime.enabled: false`; model and persisted-turn readiness false). The routing code cannot enable the endpoint by itself.

## Not present or not qualified by SC4

Mission project-status retrieval is not wired into conversational context. Project-specific privacy labels/classification and source-backed retrieval are not available as a common SIS API. Data class is a runtime setting rather than a per-turn user choice; keep chat disabled for confidential source or protected evaluation material until that boundary is explicit. No per-message model selection UI, persistent route cooldown store, specialist review delegation through SIS, or paid escalation gate is established. An owner/operator can pin one allowlisted free route in SIS config. The new `conversation` capability is the only configured route capability; OpenCode's general free route probes do not qualify coding, independent review, tool-use or large-context capabilities.

## Required acceptance before enabling chat

1. One genuine synthetic invocation through the documented OpenCode API endpoint (requires new probe authority; SC4's two CLI probes are already used).
2. Isolated SIS test with actual ChatStore commit/read-back, conversation restart, idempotent retry, and model change between turns.
3. Safe handling of provider 401/403/429/5xx, malformed output, timeout/cancellation, empty response and local fallback.
4. Owner-visible selected route/fallback reason and an explicit model pin that remains within the approved data/cost policy.
5. Independent privacy and routing review; keep production chat disabled until these pass.

Do not build the entire personal-assistant roadmap in SC4. Mission/Worker/Praxis governance responsibilities remain as documented in `CONVERSATIONAL-ARCHITECTURE.md`.
