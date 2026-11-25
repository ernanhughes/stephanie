# OpenCode Free compatibility

## Qualified invocations

On 2026-11-25, the installed OpenCode v2.3.2 CLI listed provider-qualified routes, including `opencode/deepseek-v4.1-flash-free` and `opencode/hy4-flash-free`. Each was invoked once with synthetic text; both returned the expected marker at $0. See `OPENCODE-FREE-QUALIFICATION.md` for usage and limits. No source or personal conversation was sent; no additional invocation was made.

## Supported application interface

The official OpenCode Console model reference explicitly lists OpenAI-compatible application endpoints for these chat models and says the corresponding SDK packages can call those endpoints directly. Stephanie therefore uses its existing LiteLLM transport at `https://opencode.ai/inference/openai/v1`, with the Console API key, rather than invoking the stateful CLI per conversation turn. This is a documented supported interface, not proof that the local key/route can complete an API request: direct API compatibility is not live-qualified in SC4 because both allowed probes were used through CLI.

The official reference states zero retention/no model training for the selected DeepSeek V4.1 Flash Free and Hy4 Flash Free provider routes at the time of review. Terms and zero-retention status are time-sensitive; recheck before enabling chat. Other free offerings have different privacy terms. [OpenCode Console model reference](https://opencode.ai/docs/console/models/)

The direct API route avoids OpenCode CLI archived-session persistence in addition to using Stephanie's existing ChatStore. No transcript mirror or second conversational store was added. OpenCode Go/subscription routes and metered models are separate and are not configured as automatic fallback.

## Limits

The two synthetic CLI probes qualify those exact CLI routes only. They do not establish direct API authentication, future quota, coding/review/tool-use support, model context limits, regional access, or latency. Runtime catalog visibility is not route availability. No quota values are invented, no paid call occurred, and no route was queried for quota discovery.
