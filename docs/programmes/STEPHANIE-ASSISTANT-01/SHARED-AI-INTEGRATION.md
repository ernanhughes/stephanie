# Shared AI integration

`C:\Projects\ai` contains a provider-agnostic `InferenceService` for bounded, traced reasoning and a separate namespace-scoped `ai.chat.ChatService`. The shared chat service owns a separate conversation repository/database. Its `OpenCodeCLIReasoningProvider` is reasoning-only and documents CLI route selection and failure evidence.

Stephanie already has a compatible conversational boundary: SIS calls `stephanie.services.llm_service.LLMService.achat`, which calls LiteLLM and returns model/provider/usage metadata. SA1 reused that interface and kept Stephanie-specific policy (privacy class, Mission evidence and ChatStore continuity) in SIS. It did not add another generic provider abstraction or adopt `ai.chat.ChatService`, which would create a second transcript store.

The OpenCode Go route is an opt-in route in the existing LLM call path. It uses the OpenAI-compatible Go endpoint, OpenCode Console key already configured in the environment, and a stable hashed `x-opencode-session` value derived from the Stephanie conversation ID. This was verified against OpenCode's documented Go API. No files in `C:\Projects\ai` were changed in this programme; its three pre-existing dirty files were preserved.

Future shared-AI work may expose a reusable conversational request contract and route metadata, but SA1 does not migrate Stephanie's transcript ownership or change the AI repository.
