# WorkBench integration pattern review

Reviewed `C:\Projects\workbench\src\workbench\preferences.py`, `src\workbench\service.py`, and its chat model selector/catalogue UI on 2026-11-25.

## Patterns carried into Stephanie

- A saved model choice applies to future messages and survives refresh/restart; it does not rewrite prior message attribution or conversation history.
- The model catalogue is based on what the current connection actually reports. A listed or free-priced model is not proof that this client can invoke it.
- Route state is evidence-based: unverified before a successful call, worked after a completed call, blocked after an observed refusal.
- Persist the user's preference before publishing it as active. Keep observations best-effort so a telemetry/preference-state write cannot turn a completed inference into a failed turn.

## What was not copied

Workbench uses Pi sessions and provider connections. Stephanie has an existing `LLMService`, `FreeFirstChatRouter`, and LiteLLM OpenCode-compatible transport. We retained those and did not build a second provider client, catalogue engine, or conversation store.

Stephanie adds separate consent boundaries WorkBench's chat selector does not need: message-level approval for an external turn, persistent project-category permissions, and project-context approval for each external turn. It also rejects metered routes and Ollama cloud-tagged routes in this release.
