# Model routing policy

## Intended policy

Default to deterministic mechanisms when inference is unnecessary. For eligible inference, use a capability-compatible free OpenCode route first, subject to the task's data policy. If it is unavailable, rate-limited or exhausted, consider another eligible free route. Subscription-covered, metered and local routes remain distinct alternatives governed by explicit cost and privacy policy. Never silently cross from free to paid. Preserve user override, actual/requested route, rationale, quota state, cost evidence and escalation reason. Unknown quota or cost remains unknown.

Controlled experiments must pin model, provider, route, configuration and authorized fallback before dispatch. Do not switch after a quota failure; record the failed condition or create a new protocol version.

## Current state and implementation

- Stephanie's SC3 experiment/hypothesis path remains pinned to its configured local Ollama profile and its own frozen experiment policy; free-first chat never changes that route.
- The SIS chat `LLMService.achat` now has a config-gated OpenCode free-first selector. Policy allows only `deepseek-v4.1-flash-free`, then `hy4-flash-free`, both for personal/public chat and the configured `conversation` capability. SIS itself remains disabled pending its model and persisted-turn readiness gates.
- Each free model was successfully invoked once via the local OpenCode CLI with synthetic input. The application API path is documented by OpenCode but was not invoked because the two-probe allowance had already been consumed.
- The local fallback is only the preconfigured Ollama chat route, explicitly enabled in SIS config after eligible free availability/quota failure or missing credentials. No paid/subscription route is available in this policy. Authentication/permission errors from an actual endpoint, unsupported capability, and unknown failures stop instead of silently falling through.
- Free routes require explicit privacy and capability metadata. The current `conversation` capability is the only configured capability; the policy does not yet select coding, independent review, tool use, or project-aware reasoning routes.
- Route attempts are bounded to one per eligible model per turn; 429 cooldown and provider retry-after are tracked in process memory. The local fallback cooldown when no Retry-After is present is an application cooldown, not a provider quota/reset value.
- An owner/operator may pin one of the allowlisted free routes in SIS configuration; that route is attempted alone with no automatic switch. There is not yet a per-message route-selection control in the chat UI.
- Worker route policy is separate. Stephanie does not call Worker as a general chat completion proxy. The route policy preserves the exact ChatStore message list and does not alter a frozen experiment's provider/model identity.

See `FREE-FIRST-ROUTING-IMPLEMENTATION.md` and `OPENCODE-FREE-QUALIFICATION.md` for deterministic tests and qualification gaps. Do not enable SIS until direct API and persistence acceptance plus independent review pass.
