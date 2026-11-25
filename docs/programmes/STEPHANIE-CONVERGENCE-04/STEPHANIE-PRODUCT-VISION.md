# Stephanie product vision

Stephanie is intended to become the owner's persistent conversational interface to engineering work. Mission Control remains the operational control plane for project state, approvals, execution observation and acceptance. Stephanie supplies conversation, context retrieval, reasoning, recommendations and bounded coordination through Mission's existing authority.

Her identity must survive provider changes. Chat history, user preferences, project context, objectives, operational records and reviewed experience have different owners and provenance; no provider session is the authoritative memory. Retrieved information must be labeled as retrieved context rather than model-held memory.

Longer-term capabilities include project-aware conversation, free-first task-aware routing, specialist-agent delegation through existing Worker and application interfaces, and governed actions through Mission. The complete product is roadmap-only and is not delivered by SC4.

## Current implementation findings

- Stephanie has SIS conversation routes, chat storage and existing memory/casebook mechanisms; inspect their backing configuration and retention before designating any store authoritative.
- Existing SC3/SC2 improvement code uses Stephanie's CaseBook/Experiment stores for retained experience in isolated program roots.
- Shared `ai.inference` provides provider-independent request/result/budget contracts and an OpenCode CLI provider. SC3's bounded hypothesis path currently uses local Ollama only.
- Mission is not yet a conversational host for Stephanie and does not pass selected-project context to her.
- Worker supports controlled execution and model routing but does not substitute for Mission authorization.

## Product sequence

1. Preserve and qualify existing conversation and identity.
2. Complete Mission-authorized improvement integration (SC4A–C).
3. Implement task-aware free-first model routing with explicit cost, privacy and fallback policy.
4. Add authorized project context and continuity.
5. Delegate bounded work through existing Worker/Mission/application interfaces.
6. Offer Stephanie conversation inside Mission while retaining Mission as the authority boundary.
7. Improve personalization only from reviewed experience and explicit user preferences.
